/*
 * Copyright (C) 2015, Nils Moehrle, Michael Waechter
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <numeric>
#include <algorithm>
#include <map>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#include <mve/image_color.h>
#include <acc/bvh_tree.h>
#include <Eigen/Core>
#include <Eigen/LU>

#include "util.h"
#include "histogram.h"
#include "texturing.h"
#include "sparse_table.h"
#include "progress_counter.h"

typedef acc::BVHTree<unsigned int, math::Vec3f> BVHTree;

TEX_NAMESPACE_BEGIN

/**
 * Dampens the quality of all views in which the face's projection
 * has a much different color than in the majority of views.
 * Returns whether the outlier removal was successfull.
 *
 * @param infos contains information about one face seen from several views
 * @param settings runtime configuration.
 */
bool
photometric_outlier_detection(std::vector<FaceProjectionInfo> * infos, Settings const & settings) {
    if (infos->size() == 0) return true;

    /* Configuration variables. */

    double const gauss_rejection_threshold = 6e-3;

    /* If all covariances drop below this we stop outlier detection. */
    double const minimal_covariance = 5e-4;

    int const outlier_detection_iterations = 10;
    int const minimal_num_inliers = 4;

    float outlier_removal_factor = std::numeric_limits<float>::signaling_NaN();
    switch (settings.outlier_removal) {
        case OUTLIER_REMOVAL_NONE: return true;
        case OUTLIER_REMOVAL_GAUSS_CLAMPING:
            outlier_removal_factor = 1.0f;
        break;
        case OUTLIER_REMOVAL_GAUSS_DAMPING:
            outlier_removal_factor = 0.2f;
        break;
    }

    Eigen::MatrixX3d inliers(infos->size(), 3);
    std::vector<std::uint32_t> is_inlier(infos->size(), 1);
    for (std::size_t row = 0; row < infos->size(); ++row) {
        inliers.row(row) = mve_to_eigen(infos->at(row).mean_color).cast<double>();
    }

    Eigen::RowVector3d var_mean;
    Eigen::Matrix3d covariance;
    Eigen::Matrix3d covariance_inv;

    for (int i = 0; i < outlier_detection_iterations; ++i) {

        if (inliers.rows() < minimal_num_inliers) {
            return false;
        }

        /* Calculate the inliers' mean color and color covariance. */
        var_mean = inliers.colwise().mean();
        Eigen::MatrixX3d centered = inliers.rowwise() - var_mean;
        covariance = (centered.adjoint() * centered) / double(inliers.rows() - 1);

        /* If all covariances are very small we stop outlier detection
         * and only keep the inliers (set quality of outliers to zero). */
        if (covariance.array().abs().maxCoeff() < minimal_covariance) {
            for (std::size_t row = 0; row < infos->size(); ++row) {
                if (!is_inlier[row]) infos->at(row).quality = 0.0f;
            }
            return true;
        }

        /* Invert the covariance. FullPivLU is not the fastest way but
         * it gives feedback about numerical stability during inversion. */
        Eigen::FullPivLU<Eigen::Matrix3d> lu(covariance);
        if (!lu.isInvertible()) {
            return false;
        }
        covariance_inv = lu.inverse();

        /* Compute new number of inliers (all views with a gauss value above a threshold). */
        for (std::size_t row = 0; row < infos->size(); ++row) {
            Eigen::RowVector3d color = mve_to_eigen(infos->at(row).mean_color).cast<double>();
            double gauss_value = multi_gauss_unnormalized<double, 3>(color, var_mean, covariance_inv);
            is_inlier[row] = (gauss_value >= gauss_rejection_threshold ? 1 : 0);
        }
        /* Resize Eigen matrix accordingly and fill with new inliers. */
        inliers.resize(std::accumulate(is_inlier.begin(), is_inlier.end(), 0), Eigen::NoChange);
        for (std::size_t row = 0, inlier_row = 0; row < infos->size(); ++row) {
            if (is_inlier[row]) {
                inliers.row(inlier_row++) = mve_to_eigen(infos->at(row).mean_color).cast<double>();
            }
        }
    }

    covariance_inv *= outlier_removal_factor;
    for (FaceProjectionInfo & info : *infos) {
        Eigen::RowVector3d color = mve_to_eigen(info.mean_color).cast<double>();
        double gauss_value = multi_gauss_unnormalized<double, 3>(color, var_mean, covariance_inv);
        assert(0.0 <= gauss_value && gauss_value <= 1.0);
        switch(settings.outlier_removal) {
            case OUTLIER_REMOVAL_NONE: return true;
            case OUTLIER_REMOVAL_GAUSS_DAMPING:
                info.quality *= gauss_value;
            break;
            case OUTLIER_REMOVAL_GAUSS_CLAMPING:
                if (gauss_value < gauss_rejection_threshold) info.quality = 0.0f;
            break;
        }
    }
    return true;
}

void
calculate_face_projection_infos(mve::TriangleMesh::ConstPtr mesh,
    std::vector<TextureView> * texture_views, Settings const & settings,
    FaceProjectionInfos * face_projection_infos) {

    std::vector<unsigned int> const & faces = mesh->get_faces();
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();
    mve::TriangleMesh::NormalList const & face_normals = mesh->get_face_normals();

    std::size_t const num_views = texture_views->size();

    util::WallTimer timer;
    std::cout << "\tBuilding BVH from " << faces.size() / 3 << " faces... " << std::flush;
    BVHTree bvh_tree(faces, vertices);
    std::cout << "done. (Took: " << timer.get_elapsed() << " ms)" << std::endl;

    ProgressCounter view_counter("\tCalculating face qualities", num_views);
    #pragma omp parallel
    {
        std::vector<std::pair<std::size_t, FaceProjectionInfo> > projected_face_view_infos;

        #pragma omp for schedule(dynamic)
#if !defined(_MSC_VER)
        for (std::uint16_t j = 0; j < static_cast<std::uint16_t>(num_views); ++j) {
#else
        for (std::int32_t j = 0; j < num_views; ++j) {
#endif
            view_counter.progress<SIMPLE>();

            TextureView * texture_view = &texture_views->at(j);
            texture_view->load_image();
            if (texture_view->get_image()->get_type() == mve::IMAGE_TYPE_FLOAT){
               texture_view->generate_validity_mask<float>();
            }else if (texture_view->get_image()->get_type() == mve::IMAGE_TYPE_UINT16){
               texture_view->generate_validity_mask<uint16_t>();
            }else{
               texture_view->generate_validity_mask<uint8_t>();
            }

            if (settings.data_term == DATA_TERM_GMI) {
                if (texture_view->get_image()->get_type() == mve::IMAGE_TYPE_FLOAT){
                    texture_view->generate_gradient_magnitude<float>();
                    texture_view->erode_validity_mask();
                }else if (texture_view->get_image()->get_type() == mve::IMAGE_TYPE_UINT16){
                    texture_view->generate_gradient_magnitude<uint16_t>();
                    texture_view->erode_validity_mask();
                }else{
                    texture_view->generate_gradient_magnitude<uint8_t>();
                    texture_view->erode_validity_mask();
                }
            }else if (settings.nadir_mode){
                texture_view->erode_validity_mask();
            }

            math::Vec3f const & view_pos = texture_view->get_pos();
            math::Vec3f const & viewing_direction = texture_view->get_viewing_direction();

            for (std::size_t i = 0; i < faces.size(); i += 3) {
                std::size_t face_id = i / 3;

                math::Vec3f const & v1 = vertices[faces[i]];
                math::Vec3f const & v2 = vertices[faces[i + 1]];
                math::Vec3f const & v3 = vertices[faces[i + 2]];
                math::Vec3f const & face_normal = face_normals[face_id];
                math::Vec3f const face_center = (v1 + v2 + v3) / 3.0f;

                /* Check visibility and compute quality */

                math::Vec3f view_to_face_vec = (face_center - view_pos).normalized();
                math::Vec3f face_to_view_vec = (view_pos - face_center).normalized();
                math::Vec3f up(0, 0, 1);

                /* Skip culling if the face is close to vertical
                   and 2.5D mode is enabled */
                bool vertical = false;
                if (settings.nadir_mode && fabs(up.dot(face_normal)) < 0.5) vertical = true;
                
                if (!vertical){
                    /* Backface and basic frustum culling */
                    float viewing_angle = face_to_view_vec.dot(face_normal);
                    if (viewing_angle < 0.0f || viewing_direction.dot(view_to_face_vec) < 0.0f)
                        continue;

                    if (std::acos(viewing_angle) > MATH_DEG2RAD(settings.nadir_mode ? 90.0f : 75.0f))
                        continue;
                }

                /* Projects into the valid part of the TextureView? */
                if (!texture_view->inside(v1, v2, v3))
                    continue;

                bool visible = true;
                if (!vertical && settings.geometric_visibility_test) {
                    /* Viewing rays do not collide? */
                   
                    math::Vec3f const * samples[] = {&v1, &v2, &v3};
                    // TODO: random monte carlo samples...

                    for (std::size_t k = 0; k < sizeof(samples) / sizeof(samples[0]); ++k) {
                        BVHTree::Ray ray;
                        ray.origin = *samples[k];
                        ray.dir = view_pos - ray.origin;
                        ray.tmax = ray.dir.norm();
                        ray.tmin = ray.tmax * 0.0001f;
                        ray.dir.normalize();

                        BVHTree::Hit hit;
                        if (bvh_tree.intersect(ray, &hit)) {
                            visible = false;
                            break;
                        }
                    }
                    if (!visible && !settings.nadir_mode) continue;
                }

                FaceProjectionInfo info = {j, 0.0f, math::Vec3f(0.0f, 0.0f, 0.0f)};

                /* Calculate quality. */
                if (texture_view->get_image()->get_type() == mve::IMAGE_TYPE_FLOAT){
                    texture_view->get_face_info<float>(v1, v2, v3, &info, settings);
                }else if (texture_view->get_image()->get_type() == mve::IMAGE_TYPE_UINT16){
                    texture_view->get_face_info<uint16_t>(v1, v2, v3, &info, settings);
                }else{
                    texture_view->get_face_info<uint8_t>(v1, v2, v3, &info, settings);
                }

                if (info.quality == 0.0) continue;

                if (settings.nadir_mode){
                    float m1 = up.dot(face_to_view_vec);
                    float m2 = up.dot(viewing_direction);
                    float score = (m1*m1)*(m1*m1)*(m2*m2);
                    if (!visible) info.quality = 1.0f * score;
                    else{
                        info.quality = std::max(1.0f, 10000.0f * score);
                    }
                }

                /* Change color space. */
                mve::image::color_rgb_to_ycbcr(*(info.mean_color));

                std::pair<std::size_t, FaceProjectionInfo> pair(face_id, info);
                projected_face_view_infos.push_back(pair);
            }

            texture_view->release_image();
            texture_view->release_validity_mask();
            if (settings.data_term == DATA_TERM_GMI) {
                texture_view->release_gradient_magnitude();
            }
            view_counter.inc();
        }

        //std::sort(projected_face_view_infos.begin(), projected_face_view_infos.end());

        #pragma omp critical
        {
            for (std::size_t i = projected_face_view_infos.size(); 0 < i; --i) {
                std::size_t face_id = projected_face_view_infos[i - 1].first;
                FaceProjectionInfo const & info = projected_face_view_infos[i - 1].second;
                face_projection_infos->at(face_id).push_back(info);
            }
            projected_face_view_infos.clear();
        }
    }
}

void
postprocess_face_infos(Settings const & settings,
        FaceProjectionInfos * face_projection_infos,
        DataCosts * data_costs) {

    ProgressCounter face_counter("\tPostprocessing face infos",
        face_projection_infos->size());
    #pragma omp parallel for schedule(dynamic)
#if !defined(_MSC_VER)
    for (std::size_t i = 0; i < face_projection_infos->size(); ++i) {
#else
    for (std::int64_t i = 0; i < face_projection_infos->size(); ++i) {
#endif
        face_counter.progress<SIMPLE>();

        std::vector<FaceProjectionInfo> & infos = face_projection_infos->at(i);
        if (settings.outlier_removal != OUTLIER_REMOVAL_NONE) {
            photometric_outlier_detection(&infos, settings);

            infos.erase(std::remove_if(infos.begin(), infos.end(),
                [](FaceProjectionInfo const & info) -> bool {return info.quality == 0.0f;}),
                infos.end());
        }
        std::sort(infos.begin(), infos.end());

        face_counter.inc();
    }

    /* Determine the function for the normlization. */
    float max_quality = 0.0f;
    for (std::size_t i = 0; i < face_projection_infos->size(); ++i)
        for (FaceProjectionInfo const & info : face_projection_infos->at(i))
            max_quality = std::max(max_quality, info.quality);

    Histogram hist_qualities(0.0f, max_quality, 10000);
    for (std::size_t i = 0; i < face_projection_infos->size(); ++i)
        for (FaceProjectionInfo const & info : face_projection_infos->at(i))
            hist_qualities.add_value(info.quality);
    float percentile = hist_qualities.get_approx_percentile(0.995f);

    /* Calculate the costs. */
    for (std::uint32_t i = 0; i < face_projection_infos->size(); ++i) {
        for (FaceProjectionInfo const & info : face_projection_infos->at(i)) {

            /* Clamp to percentile and normalize. */
            float normalized_quality = std::min(1.0f, info.quality / percentile);
            float data_cost = (1.0f - normalized_quality);
            data_costs->set_value(i, info.view_id, data_cost);
        }

        /* Ensure that all memory is freeed. */
        face_projection_infos->at(i) = std::vector<FaceProjectionInfo>();
    }

    std::cout << "\tMaximum quality of a face within an image: " << max_quality << std::endl;
    std::cout << "\tClamping qualities to " << percentile << " within normalization." << std::endl;
}

/* PATCH4: a square of the forced views file. */
struct ForcedSquare {
    std::string label;
    std::string image_name;
    int view_id;
    float xmin, ymin, xmax, ymax;
};

/**
 * PATCH4: reads the forced views file, one square per line: label image_name xmin ymin xmax ymax
 * (mesh coordinates, '#' starts a comment line). image_name is matched against the view's image file name,
 * with or without the extension the undistorted image adds (282-01_x.JPG matches 282-01_x.JPG.tif).
 */
std::vector<ForcedSquare>
read_forced_squares(std::string const & filename, std::vector<TextureView> const & texture_views) {
    std::vector<ForcedSquare> squares;
    std::ifstream in(filename.c_str());
    if (!in.good()) {
        std::cout << "\t[PATCH4] Cannot read the forced views file " << filename << std::endl;
        return squares;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        ForcedSquare square;
        if (!(fields >> square.label >> square.image_name >> square.xmin >> square.ymin >> square.xmax >> square.ymax)) {
            std::cout << "\t[PATCH4] Ignoring malformed line: " << line << std::endl;
            continue;
        }

        square.view_id = -1;
        std::string const prefix = square.image_name + ".";
        for (std::size_t j = 0; j < texture_views.size(); ++j) {
            std::string const & path = texture_views[j].get_image_file();
            std::size_t const slash = path.find_last_of("/\\");
            std::string const name = (slash == std::string::npos) ? path : path.substr(slash + 1);
            if (name == square.image_name || name.compare(0, prefix.size(), prefix) == 0) {
                square.view_id = static_cast<int>(j);
                break;
            }
        }
        if (square.view_id < 0) {
            std::cout << "\t[PATCH4] " << square.label << ": " << square.image_name
                << " is not one of the views, left as is" << std::endl;
            continue;
        }
        squares.push_back(square);
    }
    return squares;
}

/**
 * PATCH4: a face whose centre lies inside a square keeps that square's view as its only candidate, if the view
 * sees it, so neither the photometric outlier removal nor the view selection can choose another view there.
 * Where squares overlap, the square whose centre is nearest to the face wins.
 */
void
force_views(mve::TriangleMesh::ConstPtr mesh, std::vector<TextureView> const & texture_views,
    std::string const & filename, FaceProjectionInfos * face_projection_infos) {
    std::vector<ForcedSquare> const squares = read_forced_squares(filename, texture_views);
    if (squares.empty()) return;

    std::vector<unsigned int> const & faces = mesh->get_faces();
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();
    std::vector<std::size_t> num_forced(squares.size(), 0);
    std::vector<std::size_t> num_unseen(squares.size(), 0);

    for (std::size_t i = 0; i < faces.size(); i += 3) {
        math::Vec3f const center = (vertices[faces[i]] + vertices[faces[i + 1]] + vertices[faces[i + 2]]) / 3.0f;

        int best = -1;
        float best_distance = 0.0f;
        for (std::size_t s = 0; s < squares.size(); ++s) {
            ForcedSquare const & square = squares[s];
            if (center[0] < square.xmin || center[0] > square.xmax || center[1] < square.ymin || center[1] > square.ymax)
                continue;
            float const dx = center[0] - (square.xmin + square.xmax) / 2.0f;
            float const dy = center[1] - (square.ymin + square.ymax) / 2.0f;
            float const distance = dx * dx + dy * dy;
            if (best < 0 || distance < best_distance) {
                best = static_cast<int>(s);
                best_distance = distance;
            }
        }
        if (best < 0) continue;

        std::vector<FaceProjectionInfo> & infos = face_projection_infos->at(i / 3);
        int const view_id = squares[best].view_id;
        auto const it = std::find_if(infos.begin(), infos.end(),
            [view_id](FaceProjectionInfo const & info) -> bool {return static_cast<int>(info.view_id) == view_id;});
        if (it == infos.end()) {
            num_unseen[best] += 1;
            continue;
        }
        FaceProjectionInfo const forced = *it;
        infos.assign(1, forced);
        num_forced[best] += 1;
    }

    for (std::size_t s = 0; s < squares.size(); ++s) {
        std::cout << "\t[PATCH4] " << squares[s].label << ": " << squares[s].image_name << " forced on "
            << num_forced[s] << " faces, " << num_unseen[s] << " faces of the square not seen by it" << std::endl;
    }
}

/* PATCH7: a box of the masked boxes file, in pixels of the view's image. */
struct MaskedBox {
    float x0, y0, x1, y1;
};

/**
 * PATCH7: reads the masked boxes file, one box per line: image_name x0 y0 x1 y1, more fields ignored ('#' starts a
 * comment line). image_name is matched against the view's image file name as in read_forced_squares. Returns the boxes
 * of every view (same index as texture_views).
 */
std::vector<std::vector<MaskedBox> >
read_masked_boxes(std::string const & filename, std::vector<TextureView> const & texture_views) {
    std::vector<std::vector<MaskedBox> > boxes(texture_views.size());
    std::ifstream in(filename.c_str());
    if (!in.good()) {
        std::cout << "\t[PATCH7] Cannot read the masked boxes file " << filename << std::endl;
        return boxes;
    }

    std::map<std::string, std::size_t> by_name;
    for (std::size_t j = 0; j < texture_views.size(); ++j) {
        std::string const & path = texture_views[j].get_image_file();
        std::size_t const slash = path.find_last_of("/\\");
        by_name[(slash == std::string::npos) ? path : path.substr(slash + 1)] = j;
    }

    std::size_t num_boxes = 0;
    std::size_t num_unknown = 0;
    std::string last_name;
    long last_view = -1;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        std::string name;
        MaskedBox box;
        if (!(fields >> name >> box.x0 >> box.y0 >> box.x1 >> box.y1)) {
            std::cout << "\t[PATCH7] Ignoring malformed line: " << line << std::endl;
            continue;
        }
        if (name != last_name) {
            last_name = name;
            last_view = -1;
            std::map<std::string, std::size_t>::const_iterator const it = by_name.find(name);
            if (it != by_name.end()) {
                last_view = static_cast<long>(it->second);
            } else {
                std::string const prefix = name + ".";
                for (std::map<std::string, std::size_t>::const_iterator e = by_name.begin(); e != by_name.end(); ++e) {
                    if (e->first.compare(0, prefix.size(), prefix) == 0) {
                        last_view = static_cast<long>(e->second);
                        break;
                    }
                }
            }
        }
        if (last_view < 0) {
            num_unknown += 1;
            continue;
        }
        boxes[last_view].push_back(box);
        num_boxes += 1;
    }

    std::size_t num_views = 0;
    for (std::size_t j = 0; j < boxes.size(); ++j)
        if (!boxes[j].empty()) num_views += 1;
    std::cout << "\t[PATCH7] " << num_boxes << " boxes in " << num_views << " of " << texture_views.size() << " views";
    if (num_unknown > 0) std::cout << " (" << num_unknown << " boxes of images that are not views, left out)";
    std::cout << std::endl;
    return boxes;
}

/**
 * PATCH7: a view in which the face's projection touches a box is dropped as a candidate for that face, as long as at
 * least one view without a box sees the face; a face that every view sees with a box keeps all its views (a vehicle
 * parked during all flights stays). Runs after force_views: a forced face has one candidate, which is always kept.
 */
void
mask_views(mve::TriangleMesh::ConstPtr mesh, std::vector<TextureView> const & texture_views,
    std::string const & filename, FaceProjectionInfos * face_projection_infos) {
    std::vector<std::vector<MaskedBox> > const boxes = read_masked_boxes(filename, texture_views);

    std::vector<unsigned int> const & faces = mesh->get_faces();
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();
    std::int64_t const num_faces = static_cast<std::int64_t>(face_projection_infos->size());

    long long faces_cleared = 0;
    long long faces_all_boxed = 0;
    long long views_dropped = 0;

    #pragma omp parallel for schedule(dynamic, 1024) reduction(+:faces_cleared,faces_all_boxed,views_dropped)
    for (std::int64_t f = 0; f < num_faces; ++f) {
        std::vector<FaceProjectionInfo> & infos = face_projection_infos->at(f);
        if (infos.empty()) continue;

        math::Vec3f const & v1 = vertices[faces[f * 3]];
        math::Vec3f const & v2 = vertices[faces[f * 3 + 1]];
        math::Vec3f const & v3 = vertices[faces[f * 3 + 2]];

        std::vector<char> boxed(infos.size(), 0);
        std::size_t num_boxed = 0;
        for (std::size_t k = 0; k < infos.size(); ++k) {
            std::vector<MaskedBox> const & view_boxes = boxes[infos[k].view_id];
            if (view_boxes.empty()) continue;

            TextureView const & view = texture_views[infos[k].view_id];
            math::Vec2f const p1 = view.get_pixel_coords(v1);
            math::Vec2f const p2 = view.get_pixel_coords(v2);
            math::Vec2f const p3 = view.get_pixel_coords(v3);
            float const xmin = std::min(p1[0], std::min(p2[0], p3[0]));
            float const xmax = std::max(p1[0], std::max(p2[0], p3[0]));
            float const ymin = std::min(p1[1], std::min(p2[1], p3[1]));
            float const ymax = std::max(p1[1], std::max(p2[1], p3[1]));

            for (std::size_t b = 0; b < view_boxes.size(); ++b) {
                MaskedBox const & box = view_boxes[b];
                if (xmax >= box.x0 && xmin <= box.x1 && ymax >= box.y0 && ymin <= box.y1) {
                    boxed[k] = 1;
                    num_boxed += 1;
                    break;
                }
            }
        }

        if (num_boxed == 0) continue;
        if (num_boxed == infos.size()) {
            faces_all_boxed += 1;
            continue;
        }

        std::vector<FaceProjectionInfo> kept;
        kept.reserve(infos.size() - num_boxed);
        for (std::size_t k = 0; k < infos.size(); ++k)
            if (!boxed[k]) kept.push_back(infos[k]);
        infos.swap(kept);
        faces_cleared += 1;
        views_dropped += static_cast<long long>(num_boxed);
    }

    std::cout << "\t[PATCH7] " << faces_cleared << " faces painted only from views without a vehicle or person on them ("
        << views_dropped << " views left out), " << faces_all_boxed
        << " faces with one in every view that sees them, left as they are" << std::endl;
}

/* PATCH8: overhead wire masks per view and tile, in mesh coordinates (written by ODX's wire_masks.py). */
struct GroundMasks {
    double x0, y0, tile, cell;
    int nx, ny, n;
    /* per tile (ty * nx + tx): (view index into texture_views, offset of its mask in bits), sorted by view */
    std::vector<std::vector<std::pair<int, std::size_t> > > tiles;
    std::vector<unsigned char> bits;
};

template <typename T>
bool read_value(std::ifstream & in, T * value) {
    in.read(reinterpret_cast<char *>(value), sizeof(T));
    return static_cast<bool>(in);
}

/**
 * PATCH8: reads the wire masks file: magic HAMWIRE1, x0 y0 tile cell (doubles), nx ny n (int32), the view names, then
 * per mask: tile x, tile y, view (int32) and n * n bits (row 0 = north, numpy packbits order). Views are matched by
 * image name as in read_forced_squares. Returns false if the file cannot be read.
 */
bool
read_ground_masks(std::string const & filename, std::vector<TextureView> const & texture_views, GroundMasks * g) {
    std::ifstream in(filename.c_str(), std::ios::binary);
    char magic[8];
    if (!in.good() || !in.read(magic, 8) || std::string(magic, 8) != "HAMWIRE1") {
        std::cout << "\t[PATCH8] Cannot read the wire masks file " << filename << std::endl;
        return false;
    }
    std::int32_t nx, ny, n, num_names;
    if (!read_value(in, &g->x0) || !read_value(in, &g->y0) || !read_value(in, &g->tile) || !read_value(in, &g->cell)
        || !read_value(in, &nx) || !read_value(in, &ny) || !read_value(in, &n) || !read_value(in, &num_names)) {
        std::cout << "\t[PATCH8] The wire masks file is damaged: " << filename << std::endl;
        return false;
    }
    g->nx = nx; g->ny = ny; g->n = n;

    std::map<std::string, std::size_t> by_name;
    for (std::size_t j = 0; j < texture_views.size(); ++j) {
        std::string const & path = texture_views[j].get_image_file();
        std::size_t const slash = path.find_last_of("/\\");
        by_name[(slash == std::string::npos) ? path : path.substr(slash + 1)] = j;
    }
    std::vector<int> view_of(num_names, -1);
    for (std::int32_t k = 0; k < num_names; ++k) {
        std::uint16_t len;
        if (!read_value(in, &len)) return false;
        std::string name(len, ' ');
        if (len > 0 && !in.read(&name[0], len)) return false;
        std::map<std::string, std::size_t>::const_iterator const it = by_name.find(name);
        if (it != by_name.end()) {
            view_of[k] = static_cast<int>(it->second);
        } else {
            std::string const prefix = name + ".";
            for (std::map<std::string, std::size_t>::const_iterator e = by_name.begin(); e != by_name.end(); ++e) {
                if (e->first.compare(0, prefix.size(), prefix) == 0) {
                    view_of[k] = static_cast<int>(e->second);
                    break;
                }
            }
        }
    }

    std::int64_t num_masks;
    if (!read_value(in, &num_masks)) return false;
    std::size_t const mask_bytes = (static_cast<std::size_t>(n) * n + 7) / 8;
    g->tiles.assign(static_cast<std::size_t>(nx) * ny, std::vector<std::pair<int, std::size_t> >());
    g->bits.resize(static_cast<std::size_t>(num_masks) * mask_bytes);
    std::size_t used = 0, unknown = 0;
    for (std::int64_t m = 0; m < num_masks; ++m) {
        std::int32_t tx, ty, k;
        if (!read_value(in, &tx) || !read_value(in, &ty) || !read_value(in, &k)) return false;
        if (!in.read(reinterpret_cast<char *>(&g->bits[used * mask_bytes]), mask_bytes)) return false;
        if (k < 0 || k >= num_names || view_of[k] < 0 || tx < 0 || ty < 0 || tx >= nx || ty >= ny) {
            unknown += 1;
            continue;
        }
        g->tiles[static_cast<std::size_t>(ty) * nx + tx].push_back(std::make_pair(view_of[k], used * mask_bytes));
        used += 1;
    }
    g->bits.resize(used * mask_bytes);
    for (std::size_t t = 0; t < g->tiles.size(); ++t)
        std::sort(g->tiles[t].begin(), g->tiles[t].end());

    std::cout << "\t[PATCH8] " << used << " wire masks read (" << nx << " x " << ny << " tiles of " << g->tile << " m)";
    if (unknown > 0) std::cout << ", " << unknown << " of images that are not views left out";
    std::cout << std::endl;
    return true;
}

/**
 * PATCH8: for each face, the views whose mask has a wire under the face, and the views not checked there, are dropped
 * as candidates, as long as at least one view checked there has no wire under it; otherwise the face keeps its views.
 * Runs after force_views and mask_views.
 */
void
mask_ground(mve::TriangleMesh::ConstPtr mesh, std::vector<TextureView> const & texture_views,
    std::string const & filename, FaceProjectionInfos * face_projection_infos) {
    GroundMasks g;
    if (!read_ground_masks(filename, texture_views, &g)) return;

    std::vector<unsigned int> const & faces = mesh->get_faces();
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();
    std::int64_t const num_faces = static_cast<std::int64_t>(face_projection_infos->size());
    int const max_cells = 64;

    long long faces_cleared = 0;
    long long faces_without_clean = 0;
    long long views_dropped = 0;

    #pragma omp parallel for schedule(dynamic, 1024) reduction(+:faces_cleared,faces_without_clean,views_dropped)
    for (std::int64_t f = 0; f < num_faces; ++f) {
        std::vector<FaceProjectionInfo> & infos = face_projection_infos->at(f);
        if (infos.empty()) continue;

        math::Vec3f const & v1 = vertices[faces[f * 3]];
        math::Vec3f const & v2 = vertices[faces[f * 3 + 1]];
        math::Vec3f const & v3 = vertices[faces[f * 3 + 2]];
        double const xmin = std::min(v1[0], std::min(v2[0], v3[0]));
        double const xmax = std::max(v1[0], std::max(v2[0], v3[0]));
        double const ymin = std::min(v1[1], std::min(v2[1], v3[1]));
        double const ymax = std::max(v1[1], std::max(v2[1], v3[1]));
        long gx0 = static_cast<long>(std::floor((xmin - g.x0) / g.cell));
        long gx1 = static_cast<long>(std::floor((xmax - g.x0) / g.cell));
        long gy0 = static_cast<long>(std::floor((ymin - g.y0) / g.cell));
        long gy1 = static_cast<long>(std::floor((ymax - g.y0) / g.cell));
        if (gx1 - gx0 >= max_cells) gx1 = gx0 + max_cells - 1;
        if (gy1 - gy0 >= max_cells) gy1 = gy0 + max_cells - 1;
        if (gx1 < 0 || gy1 < 0 || gx0 >= static_cast<long>(g.nx) * g.n || gy0 >= static_cast<long>(g.ny) * g.n) continue;

        /* 0 clean, 1 wire, 2 not checked there */
        std::vector<char> state(infos.size(), 0);
        std::size_t num_clean = 0;
        for (std::size_t k = 0; k < infos.size(); ++k) {
            int const view = static_cast<int>(infos[k].view_id);
            char st = 0;
            long last_tile = -1;
            std::size_t offset = 0;
            bool have = false;
            for (long gy = gy0; gy <= gy1 && st != 2; ++gy) {
                for (long gx = gx0; gx <= gx1; ++gx) {
                    if (gx < 0 || gy < 0 || gx >= static_cast<long>(g.nx) * g.n || gy >= static_cast<long>(g.ny) * g.n) { st = 2; break; }
                    long const tx = gx / g.n, ty = gy / g.n;
                    long const tile = ty * g.nx + tx;
                    if (tile != last_tile) {
                        last_tile = tile;
                        std::vector<std::pair<int, std::size_t> > const & t = g.tiles[tile];
                        std::vector<std::pair<int, std::size_t> >::const_iterator const it =
                            std::lower_bound(t.begin(), t.end(), std::make_pair(view, static_cast<std::size_t>(0)));
                        have = (it != t.end() && it->first == view);
                        if (have) offset = it->second;
                    }
                    if (!have) { st = 2; break; }
                    long const c = gx - tx * g.n;
                    long const r = g.n - 1 - (gy - ty * g.n);
                    std::size_t const bit = static_cast<std::size_t>(r) * g.n + static_cast<std::size_t>(c);
                    if ((g.bits[offset + (bit >> 3)] >> (7 - (bit & 7))) & 1) st = 1;
                }
            }
            state[k] = st;
            if (st == 0) num_clean += 1;
        }

        if (num_clean == infos.size()) continue;
        if (num_clean == 0) {
            bool any_checked = false;
            for (std::size_t k = 0; k < infos.size(); ++k) if (state[k] == 1) any_checked = true;
            if (any_checked) faces_without_clean += 1;
            continue;
        }

        std::vector<FaceProjectionInfo> kept;
        kept.reserve(num_clean);
        for (std::size_t k = 0; k < infos.size(); ++k)
            if (state[k] == 0) kept.push_back(infos[k]);
        views_dropped += static_cast<long long>(infos.size() - num_clean);
        infos.swap(kept);
        faces_cleared += 1;
    }

    std::cout << "\t[PATCH8] " << faces_cleared << " faces painted only from views without a wire over them ("
        << views_dropped << " views left out), " << faces_without_clean
        << " faces with a wire in every view checked there, left as they are" << std::endl;
}

void
calculate_data_costs(mve::TriangleMesh::ConstPtr mesh, std::vector<TextureView> * texture_views,
    Settings const & settings, DataCosts * data_costs) {

    std::size_t const num_faces = mesh->get_faces().size() / 3;
    std::size_t const num_views = texture_views->size();

    if (num_faces > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("Exeeded maximal number of faces");
    if (num_views > std::numeric_limits<std::uint16_t>::max())
        throw std::runtime_error("Exeeded maximal number of views");

    FaceProjectionInfos face_projection_infos(num_faces);
    calculate_face_projection_infos(mesh, texture_views, settings, &face_projection_infos);
    if (!settings.forced_views_file.empty()) {
        force_views(mesh, *texture_views, settings.forced_views_file, &face_projection_infos);
    }
    if (!settings.masked_boxes_file.empty()) {
        mask_views(mesh, *texture_views, settings.masked_boxes_file, &face_projection_infos);
    }
    if (!settings.masked_ground_file.empty()) {
        mask_ground(mesh, *texture_views, settings.masked_ground_file, &face_projection_infos);
    }
    postprocess_face_infos(settings, &face_projection_infos, data_costs);
}

TEX_NAMESPACE_END
