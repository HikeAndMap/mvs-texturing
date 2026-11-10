/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <tbb/task_arena.h>
#include <omp.h>

#include <util/timer.h>
#include <util/system.h>
#include <util/file_system.h>
#include <mve/mesh_io_ply.h>

#include "tex/util.h"
#include "tex/timer.h"
#include "tex/debug.h"
#include "tex/texturing.h"
#include "tex/progress_counter.h"
#include "tex/step_timer.h"

#include "arguments.h"

int main(int argc, char **argv) {
    util::system::print_build_timestamp(argv[0]);
    util::system::register_segfault_handler();

    Timer timer;
    util::WallTimer wtimer;
    tex::StepTimer step_timer;

    Arguments conf;
    try {
        conf = parse_args(argc, argv);
    } catch (std::invalid_argument & ia) {
        std::cerr << ia.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string const out_dir = util::fs::dirname(conf.out_prefix);

    if (!util::fs::dir_exists(out_dir.c_str())) {
        std::cerr << "Destination directory does not exist!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string const tmp_dir = util::fs::join_path(out_dir, "tmp");
    if (!util::fs::dir_exists(tmp_dir.c_str())) {
        util::fs::mkdir(tmp_dir.c_str());
    } else {
        std::cout << "Careful! Temporary directory \"tmp\" exists within the destination directory." << std::endl;
    }

    // Set the number of threads to use.
    tbb::task_arena arena(conf.num_threads > 0 ? conf.num_threads : tbb::this_task_arena::max_concurrency());
    if (conf.num_threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(conf.num_threads);
    }

    std::cout << "Load and prepare mesh: " << std::endl;
    mve::TriangleMesh::Ptr mesh;
    try {
        mesh = mve::geom::load_ply_mesh(conf.in_mesh);
    } catch (std::exception& e) {
        std::cerr << "\tCould not load mesh: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::cout << "Generating texture views: " << std::endl;
    tex::TextureViews texture_views;
    mve::ImageType type = tex::generate_texture_views(conf.in_scene, &texture_views, tmp_dir);
    step_timer.step("mesh and views loaded");

    /* PATCH8: faces under wires cut smaller first (needs the views' names to read the wire masks); prepare_mesh then
     * computes the normals and vertex infos of the mesh as it is from here on. */
    if (!conf.settings.masked_ground_file.empty() && conf.settings.wire_split_min_edge > 0.0f) {
        std::cout << "Splitting faces under wires: " << std::endl;
        tex::split_wire_faces(mesh, texture_views, conf.settings);
        step_timer.step("[PATCH8] faces under wires split");
    }
    mve::MeshInfo mesh_info(mesh);
    tex::prepare_mesh(&mesh_info, mesh);
    step_timer.step("mesh prepared");

    write_string_to_file(conf.out_prefix + ".conf", conf.to_string());
    timer.measure("Loading");

    std::size_t const num_faces = mesh->get_faces().size() / 3;

    std::cout << "Building adjacency graph: " << std::endl;
    tex::Graph graph(num_faces);
    tex::build_adjacency_graph(mesh, mesh_info, &graph);
    step_timer.step("adjacency graph");

    if (conf.labeling_file.empty()) {
        std::cout << "View selection:" << std::endl;
        util::WallTimer rwtimer;

        tex::DataCosts data_costs(num_faces, texture_views.size());
        if (conf.data_cost_file.empty()) {
            tex::calculate_data_costs(mesh, &texture_views, conf.settings, &data_costs);

            if (conf.write_intermediate_results) {
                std::cout << "\tWriting data cost file... " << std::flush;
                tex::DataCosts::save_to_file(data_costs, conf.out_prefix + "_data_costs.spt");
                std::cout << "done." << std::endl;
            }
        } else {
            std::cout << "\tLoading data cost file... " << std::flush;
            try {
                tex::DataCosts::load_from_file(conf.data_cost_file, &data_costs);
            } catch (util::FileException e) {
                std::cout << "failed!" << std::endl;
                std::cerr << e.what() << std::endl;
                std::exit(EXIT_FAILURE);
            }
            std::cout << "done." << std::endl;
        }
        timer.measure("Calculating data costs");
        step_timer.step("data costs, all together");

        try {
            tex::view_selection(data_costs, &graph, conf.settings);
        } catch (std::runtime_error& e) {
            std::cerr << "\tOptimization failed: " << e.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }
        timer.measure("Running MRF optimization");
        std::cout << "\tTook: " << rwtimer.get_elapsed_sec() << "s" << std::endl;
        step_timer.step("view selection, all together");

        /* Write labeling to file. */
        if (conf.write_intermediate_results || conf.write_labeling) {
            std::vector<std::size_t> labeling(graph.num_nodes());
            for (std::size_t i = 0; i < graph.num_nodes(); ++i) {
                labeling[i] = graph.get_label(i);
            }
            vector_to_file(conf.out_prefix + "_labeling.vec", labeling);
        }
        /* PATCH10: label k is view k - 1 (0 = none); the views' images, one per line, and every face's triangle (x, y of its
         * three corners, float32) with its label (uint32) after the magic HAMLAB01 and the face count (int64), so that ODX can
         * map which photo painted where (prepare_mesh may have removed faces, so the mesh file's face order does not do). */
        if (conf.write_labeling) {
            std::ofstream views_out((conf.out_prefix + "_labeling_views.txt").c_str());
            for (std::size_t j = 0; j < texture_views.size(); ++j)
                views_out << texture_views[j].get_image_file() << "\n";

            std::vector<unsigned int> const & mesh_faces = mesh->get_faces();
            std::vector<math::Vec3f> const & mesh_vertices = mesh->get_vertices();
            std::ofstream faces_out((conf.out_prefix + "_labeling_faces.bin").c_str(), std::ios::binary);
            std::int64_t const num_written = static_cast<std::int64_t>(graph.num_nodes());
            faces_out.write("HAMLAB01", 8);
            faces_out.write(reinterpret_cast<char const *>(&num_written), sizeof(num_written));
            for (std::size_t i = 0; i < graph.num_nodes(); ++i) {
                float xy[6];
                for (int c = 0; c < 3; ++c) {
                    math::Vec3f const & v = mesh_vertices[mesh_faces[i * 3 + c]];
                    xy[c * 2] = v[0];
                    xy[c * 2 + 1] = v[1];
                }
                std::uint32_t const label = static_cast<std::uint32_t>(graph.get_label(i));
                faces_out.write(reinterpret_cast<char const *>(xy), sizeof(xy));
                faces_out.write(reinterpret_cast<char const *>(&label), sizeof(label));
            }
        }
        if (conf.write_intermediate_results || conf.write_labeling) step_timer.step("labeling written");
    } else {
        std::cout << "Loading labeling from file... " << std::flush;

        /* Load labeling from file. */
        std::vector<std::size_t> labeling = vector_from_file<std::size_t>(conf.labeling_file);
        if (labeling.size() != graph.num_nodes()) {
            std::cerr << "Wrong labeling file for this mesh/scene combination... aborting!" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        /* Transfer labeling to graph. */
        for (std::size_t i = 0; i < labeling.size(); ++i) {
            const std::size_t label = labeling[i];
            if (label > texture_views.size()){
                std::cerr << "Wrong labeling file for this mesh/scene combination... aborting!" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            graph.set_label(i, label);
        }

        std::cout << "done." << std::endl;
    }

    tex::TextureAtlases texture_atlases;
    {
        /* Create texture patches and adjust them. */
        tex::TexturePatches texture_patches;
        tex::VertexProjectionInfos vertex_projection_infos;
        std::cout << "Generating texture patches:" << std::endl;
        tex::generate_texture_patches(graph, mesh, mesh_info, &texture_views,
            conf.settings, &vertex_projection_infos, &texture_patches);
        step_timer.step("texture patches");

        if (conf.settings.global_seam_leveling) {
            std::cout << "Running global seam leveling:" << std::endl;
            tex::global_seam_leveling(graph, mesh, mesh_info, vertex_projection_infos, &texture_patches);
            timer.measure("Running global seam leveling");
            step_timer.step("global seam leveling");
        } else {
            ProgressCounter texture_patch_counter("Calculating validity masks for texture patches", texture_patches.size());
            #pragma omp parallel for schedule(dynamic)
#if !defined(_MSC_VER)
            for (std::size_t i = 0; i < texture_patches.size(); ++i) {
#else
            for (std::int64_t i = 0; i < texture_patches.size(); ++i) {
#endif
                texture_patch_counter.progress<SIMPLE>();
                TexturePatch::Ptr texture_patch = texture_patches[i];
                std::vector<math::Vec3f> patch_adjust_values(texture_patch->get_faces().size() * 3, math::Vec3f(0.0f));
                texture_patch->adjust_colors(patch_adjust_values);
                texture_patch_counter.inc();
            }
            timer.measure("Calculating texture patch validity masks");
            step_timer.step("texture patch validity masks");
        }

        if (conf.settings.local_seam_leveling) {
            std::cout << "Running local seam leveling:" << std::endl;
            tex::local_seam_leveling(graph, mesh, vertex_projection_infos, &texture_patches);
            step_timer.step("local seam leveling");
        }
        timer.measure("Running local seam leveling");

        /* Generate texture atlases. */
        std::cout << "Generating texture atlases:" << std::endl;

        bool grayscale = texture_views.front().is_grayscale();
        tex::generate_texture_atlases(&texture_patches, conf.settings, &texture_atlases, type, grayscale);
        step_timer.step("texture atlases");
    }
    step_timer.step("texture patches freed");

    /* Create and write out obj model. */
    {
        std::cout << "Building objmodel:" << std::endl;
        tex::Model model;
        tex::build_model(mesh, texture_atlases, &model);
        timer.measure("Building OBJ model");
        step_timer.step("model built");

        std::cout << "\tSaving model... " << std::flush;
        tex::Model::save(model, conf.out_prefix);
        std::cout << "done." << std::endl;
        timer.measure("Saving");
        step_timer.step("model saved");
    }

    std::cout << "Whole texturing procedure took: " << wtimer.get_elapsed_sec() << "s" << std::endl;
    timer.measure("Total");
    if (conf.write_timings) {
        timer.write_to_file(conf.out_prefix + "_timings.csv");
    }

    if (conf.write_view_selection_model) {
        texture_atlases.clear();
        std::cout << "Generating debug texture patches:" << std::endl;
        {
            tex::TexturePatches texture_patches;
            generate_debug_embeddings(&texture_views);
            tex::VertexProjectionInfos vertex_projection_infos; // Will only be written
            tex::generate_texture_patches(graph, mesh, mesh_info, &texture_views,
                conf.settings, &vertex_projection_infos, &texture_patches);
            tex::generate_texture_atlases(&texture_patches, conf.settings, &texture_atlases, type, false);
        }

        std::cout << "Building debug objmodel:" << std::endl;
        {
            tex::Model model;
            tex::build_model(mesh, texture_atlases, &model);
            std::cout << "\tSaving model... " << std::flush;
            tex::Model::save(model, conf.out_prefix + "_view_selection");
            std::cout << "done." << std::endl;
        }
    }

    /* Remove temporary files. */
    for (util::fs::File const & file : util::fs::Directory(tmp_dir)) {
        util::fs::unlink(util::fs::join_path(file.path, file.name).c_str());
    }
    util::fs::rmdir(tmp_dir.c_str());

    return EXIT_SUCCESS;
}
