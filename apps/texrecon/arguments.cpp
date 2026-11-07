/*
 * Copyright (C) 2015, Nils Moehrle, Michael Waechter
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "arguments.h"
#include "util/file_system.h"

#define SKIP_GLOBAL_SEAM_LEVELING "skip_global_seam_leveling"
#define SKIP_GEOMETRIC_VISIBILITY_TEST "skip_geometric_visibility_test"
#define SKIP_LOCAL_SEAM_LEVELING "skip_local_seam_leveling"
#define NO_INTERMEDIATE_RESULTS "no_intermediate_results"
#define WRITE_TIMINGS "write_timings"
#define SKIP_HOLE_FILLING "skip_hole_filling"
#define KEEP_UNSEEN_FACES "keep_unseen_faces"
#define NADIR_MODE    "nadir_mode"
#define NUM_THREADS "num_threads"
#define MAX_TEXTURE_SIZE "max_texture_size"
#define FORCED_VIEWS "forced_views"
#define MASKED_BOXES "masked_boxes"
#define MASKED_GROUND "masked_ground"
#define VIEW_SURVEYS "view_surveys"
#define SURVEY_MAX_ANGLE "survey_max_angle"
#define WRITE_LABELING "write_labeling"

Arguments parse_args(int argc, char **argv) {
    util::Arguments args;
    args.set_exit_on_error(true);
    args.set_nonopt_maxnum(3);
    args.set_nonopt_minnum(3);
    args.set_helptext_indent(34);
    args.set_description("Textures a mesh given images in form of a 3D scene.");
    args.set_usage("Usage: " + std::string(argv[0]) + " [options] IN_SCENE IN_MESH OUT_PREFIX"
        "\n\nIN_SCENE := (SCENE_FOLDER | BUNDLE_FILE | MVE_SCENE::EMBEDDING)"
        "\n\nSCENE_FOLDER:"
        "\nWithin a scene folder a .cam file has to be given for each image."
        "\nA .cam file is structured as follows:"
        "\n    tx ty tz R00 R01 R02 R10 R11 R12 R20 R21 R22"
        "\n    f d0 d1 paspect ppx ppy"
        "\nFirst line: Extrinsics - translation vector and rotation matrix (the transform from world to camera)"
        "\nSecond line: Intrinsics - focal length, distortion coefficients, pixel aspect ratio and principal point"
        "\nThe focal length is the distance between camera center and image plane normalized by dividing with the larger image dimension."
        "\nFor non zero distortion coefficients the image will be undistorted prior to the texturing process."
        " If only d0 is non zero the Noah Snavely's distortion model is assumed otherwise the distortion model of VSFM is assumed."
        "\nThe pixel aspect ratio is usually 1 or close to 1. If your SfM system doesn't output it, but outputs a different focal length in x and y direction, you have to encode this here."
        "\nThe principal point has to be given in unit dimensions (e.g. 0.5 0.5)."
        "\n\nBUNDLE_FILE:"
        "\nCurrently only NVM bundle files (from VisualSFM, http://ccwu.me/vsfm/) are supported."
        "\nSince the bundle file contains relative paths to the images please make sure you did not move them (relative to the bundle) or rename them after the bundling process."
        "\n\nMVE_SCENE::EMBEDDING:"
        "\nThis is the scene representation we use in our research group: http://www.gris.tu-darmstadt.de/projects/multiview-environment/."
        "\n\nIN_MESH:"
        "\nThe mesh that you want to texture and which needs to be in the same coordinate frame as the camera parameters. You can reconstruct one, e.g. with CMVS: http://www.di.ens.fr/cmvs/"
        "\n\nOUT_PREFIX:"
        "\nA path and name for the output files, e.g. <path>/<to>/my_textured_mesh"
        "\nDon't append an obj extension. The application does that itself because it outputs multiple files (mesh, material file, texture files)."
        "\n");
    args.add_option('D',"data_cost_file", true,
        "Skip calculation of data costs and use the ones provided in the given file");
    args.add_option('L',"labeling_file", true,
        "Skip view selection and use the labeling provided in the given file");
    args.add_option('d',"data_term", true,
        "Data term: {" +
        choices<tex::DataTerm>() + "} [" +
        choice_string<tex::DataTerm>(tex::DATA_TERM_GMI) + "]");
    args.add_option('s',"smoothness_term", true,
        "Smoothness term: {" +
        choices<tex::SmoothnessTerm>() + "} [" +
        choice_string<tex::SmoothnessTerm>(tex::SMOOTHNESS_TERM_POTTS) + "]");
    args.add_option('o',"outlier_removal", true,
        "Photometric outlier (pedestrians etc.) removal method: {" +
        choices<tex::OutlierRemoval>() +  "} [" +
        choice_string<tex::OutlierRemoval>(tex::OUTLIER_REMOVAL_NONE) + "]");
    args.add_option('t',"tone_mapping", true,
        "Tone mapping method: {" +
        choices<tex::ToneMapping>() +  "} [" +
        choice_string<tex::ToneMapping>(tex::TONE_MAPPING_NONE) + "]");
    args.add_option('v',"view_selection_model", false,
        "Write out view selection model [false]");
    args.add_option('\0', SKIP_GEOMETRIC_VISIBILITY_TEST, false,
        "Skip geometric visibility test based on ray intersection [false]");
    args.add_option('\0', SKIP_GLOBAL_SEAM_LEVELING, false,
        "Skip global seam leveling [false]");
    args.add_option('\0', SKIP_LOCAL_SEAM_LEVELING, false,
        "Skip local seam leveling (Poisson editing) [false]");
    args.add_option('\0', SKIP_HOLE_FILLING, false,
        "Skip hole filling [false]");
    args.add_option('\0', KEEP_UNSEEN_FACES, false,
        "Keep unseen faces [false]");
    args.add_option('\0', NADIR_MODE, false,
        "Turn on nadir mode [false]");
    args.add_option('\0', WRITE_TIMINGS, false,
        "Write out timings for each algorithm step (OUT_PREFIX + _timings.csv)");
    args.add_option('\0', NO_INTERMEDIATE_RESULTS, false,
        "Do not write out intermediate results");
    args.add_option('\0', NUM_THREADS, true,
        "How many threads to use. Set 1 for determinism.");
    args.add_option('\0', MAX_TEXTURE_SIZE, true,
        "Maximum size of output textures.");
    args.add_option('\0', FORCED_VIEWS, true,
        "File with views forced on the faces inside given squares, one per line: "
        "label image_name xmin ymin xmax ymax (mesh coordinates)");
    args.add_option('\0', MASKED_BOXES, true,
        "File with boxes per view, one per line: image_name x0 y0 x1 y1 (pixels; more fields are ignored). "
        "A face is not painted from a view in which it touches a box, as long as a view without one sees it");
    args.add_option('\0', MASKED_GROUND, true,
        "File with wire masks per view and tile in mesh coordinates (ODX wire_masks.py). "
        "A face is not painted from a view with a wire over it, as long as a view without one sees it");
    args.add_option('\0', VIEW_SURVEYS, true,
        "File with the survey of every view, one per line: image_name rank reserve (rank 0 = the newest survey; reserve 1 = "
        "an older photo inside a newer survey). A face is painted from the newest survey that sees it within the survey "
        "angle; a reserve view only paints GCP squares and the ground under vehicles");
    args.add_option('\0', SURVEY_MAX_ANGLE, true,
        "Angle from straight down within which a survey's view counts as seeing a face [25]");
    args.add_option('\0', WRITE_LABELING, false,
        "Write the labeling (the view of every face) and the view names, also with no_intermediate_results");
    args.parse(argc, argv);

    Arguments conf;
    conf.in_scene = args.get_nth_nonopt(0);
    conf.in_mesh = args.get_nth_nonopt(1);
    conf.out_prefix = util::fs::sanitize_path(args.get_nth_nonopt(2));

    /* Set defaults for optional arguments. */
    conf.data_cost_file = "";
    conf.labeling_file = "";

    conf.write_timings = false;
    conf.write_intermediate_results = true;
    conf.write_labeling = false;
    conf.write_view_selection_model = false;

    conf.num_threads = -1;

    /* Handle optional arguments. */
    for (util::ArgResult const* i = args.next_option();
         i != 0; i = args.next_option()) {
        switch (i->opt->sopt) {
        case 'v':
            conf.write_view_selection_model = true;
        break;
        case 'D':
            conf.data_cost_file = i->arg;
        break;
        case 'L':
            conf.labeling_file = i->arg;
        break;
        case 'd':
            conf.settings.data_term = parse_choice<tex::DataTerm>(i->arg);
        break;
        case 's':
            conf.settings.smoothness_term = parse_choice<tex::SmoothnessTerm>(i->arg);
        break;
        case 'o':
            conf.settings.outlier_removal = parse_choice<tex::OutlierRemoval>(i->arg);
        break;
        case 't':
            conf.settings.tone_mapping = parse_choice<tex::ToneMapping>(i->arg);
        break;
        case '\0':
            if (i->opt->lopt == SKIP_GEOMETRIC_VISIBILITY_TEST) {
                conf.settings.geometric_visibility_test = false;
            } else if (i->opt->lopt == SKIP_GLOBAL_SEAM_LEVELING) {
                conf.settings.global_seam_leveling = false;
            } else if (i->opt->lopt == SKIP_LOCAL_SEAM_LEVELING) {
                conf.settings.local_seam_leveling = false;
            } else if (i->opt->lopt == SKIP_HOLE_FILLING) {
                conf.settings.hole_filling = false;
            } else if (i->opt->lopt == KEEP_UNSEEN_FACES) {
                conf.settings.keep_unseen_faces = true;
            } else if (i->opt->lopt == NADIR_MODE) {
                conf.settings.nadir_mode = true;
            } else if (i->opt->lopt == WRITE_TIMINGS) {
                conf.write_timings = true;
            } else if (i->opt->lopt == NO_INTERMEDIATE_RESULTS) {
                conf.write_intermediate_results = false;
            } else if (i->opt->lopt == NUM_THREADS) {
                conf.num_threads = std::stoi(i->arg);
            } else if (i->opt->lopt == MAX_TEXTURE_SIZE) {
                conf.settings.max_texture_size = std::stoi(i->arg);
            } else if (i->opt->lopt == FORCED_VIEWS) {
                conf.settings.forced_views_file = i->arg;
            } else if (i->opt->lopt == MASKED_BOXES) {
                conf.settings.masked_boxes_file = i->arg;
            } else if (i->opt->lopt == MASKED_GROUND) {
                conf.settings.masked_ground_file = i->arg;
            } else if (i->opt->lopt == VIEW_SURVEYS) {
                conf.settings.view_surveys_file = i->arg;
            } else if (i->opt->lopt == SURVEY_MAX_ANGLE) {
                conf.settings.survey_max_angle = std::stof(i->arg);
            } else if (i->opt->lopt == WRITE_LABELING) {
                conf.write_labeling = true;
            } else {
                throw std::invalid_argument("Invalid long option");
            }
        break;
        default:
            throw std::invalid_argument("Invalid short option");
        }
    }

    if (conf.settings.nadir_mode){
        // Always set to area
        conf.settings.data_term = tex::DataTerm::DATA_TERM_AREA;
    }

    return conf;
}

std::string
bool_to_string(bool b){
    return b ? "True" : "False";
}

std::string
Arguments::to_string(){
    std::stringstream out;
    out << "Input scene: \t" << in_scene << std::endl
        << "Input mesh: \t" << in_mesh << std::endl
        << "Output prefix: \t" << out_prefix << std::endl
        << "Datacost file: \t" << data_cost_file << std::endl
        << "Labeling file: \t" << labeling_file << std::endl
        << "Forced views file: \t" << settings.forced_views_file << std::endl
        << "Masked boxes file: \t" << settings.masked_boxes_file << std::endl
        << "Masked ground file: \t" << settings.masked_ground_file << std::endl
        << "View surveys file: \t" << settings.view_surveys_file << std::endl
        << "Survey max angle: \t" << settings.survey_max_angle << std::endl
        << "Data term: \t" << choice_string<tex::DataTerm>(settings.data_term) << std::endl
        << "Smoothness term: \t" << choice_string<tex::SmoothnessTerm>(settings.smoothness_term) << std::endl
        << "Outlier removal method: \t" << choice_string<tex::OutlierRemoval>(settings.outlier_removal) << std::endl
        << "Tone mapping: \t" << choice_string<tex::ToneMapping>(settings.tone_mapping) << std::endl
        << "Apply global seam leveling: \t" << bool_to_string(settings.global_seam_leveling) << std::endl
        << "Apply local seam leveling: \t" << bool_to_string(settings.local_seam_leveling) << std::endl;

    return out.str();
}
