/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef TEX_SETTINGS_HEADER
#define TEX_SETTINGS_HEADER

#include <sstream>
#include <vector>
#include <string>

#include "defines.h"

template <typename T>
const std::vector<std::string> choice_strings();

template <typename T> inline
const std::string choice_string(T i) {
    return choice_strings<T>()[static_cast<std::size_t>(i)];
}

template <typename T> inline
const std::string choices() {
    const std::vector<std::string> strings = choice_strings<T>();
    const std::size_t n = strings.size();
    std::stringstream ss;
    for (std::size_t i = 0; i < n; ++i) {
        ss << strings[i];
        if(i != n - 1) ss << ", ";
    }
    return ss.str();
}

template <typename T> inline
T parse_choice(std::string s) {
    const std::vector<std::string> strings = choice_strings<T>();
    const std::size_t n = strings.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (s == strings[i]) {
            return static_cast<T>(i);
        }
    }

    std::stringstream ss;
    ss << "Invalid choice: " << s << " (Available choices: ";
    ss << choices<T>() << ")";

    throw std::invalid_argument(ss.str());
}

TEX_NAMESPACE_BEGIN

/** Enum representing a data term. */
enum DataTerm {
    DATA_TERM_AREA = 0,
    DATA_TERM_GMI = 1
};

/** Enum representing a smoothness term. */
enum SmoothnessTerm {
    SMOOTHNESS_TERM_POTTS = 0
};

/** Enum representing outlier removal choice. */
enum OutlierRemoval {
    OUTLIER_REMOVAL_NONE = 0,
    OUTLIER_REMOVAL_GAUSS_DAMPING = 1,
    OUTLIER_REMOVAL_GAUSS_CLAMPING = 2
};

/** Enum representing tone mapping choice. */
enum ToneMapping {
    TONE_MAPPING_NONE = 0,
    TONE_MAPPING_GAMMA = 1
};

struct Settings {
    bool verbose = false;

    DataTerm data_term = DATA_TERM_GMI;
    SmoothnessTerm smoothness_term = SMOOTHNESS_TERM_POTTS;
    OutlierRemoval outlier_removal = OUTLIER_REMOVAL_NONE;
    ToneMapping tone_mapping = TONE_MAPPING_NONE;

    bool geometric_visibility_test = true;
    bool global_seam_leveling = true;
    bool local_seam_leveling = true;
    bool hole_filling = true;
    bool keep_unseen_faces = false;
    bool nadir_mode = false;

    int max_texture_size = 8 * 1024;

    /* PATCH4: file with the views forced on the faces inside given squares (empty: none). */
    std::string forced_views_file;

    /* PATCH7: file with boxes (vehicles, people) per view; a face is not painted from a view where it touches a box, as
     * long as a view without one sees it (empty: none). */
    std::string masked_boxes_file;

    /* PATCH8: file with overhead wire masks per view and tile in mesh coordinates; a face is not painted from a view with
     * a wire over it, as long as a view without one (and checked there) sees it (empty: none). */
    std::string masked_ground_file;

    /* PATCH8: with wire masks, a face with a wire in every view checked there is cut into smaller faces, again and again,
     * until one of its views is clean or its longest edge is at most this (metres). 0: faces are not split. */
    float wire_split_min_edge = 0.12f;

    /* PATCH8: a face is cut only when cutting can win at least this many blocks (about wire_split_min_edge square) that
     * are clean in some view but not in the view that would paint the whole face. */
    int wire_split_min_gain = 1;

    /* PATCH10: file with the survey of every view (rank 0 = the newest) and whether the view is a reserve view (an older
     * photo inside a newer survey: it paints only GCP squares and the ground under vehicles). Empty: one survey, as before. */
    std::string view_surveys_file;

    /* PATCH10: a survey paints a face when one of its views sees the face within this angle from straight down. */
    float survey_max_angle = 25.0f;

    /* PATCH10: a newer survey paints a face only when its most straight-down view is within the larger of these two:
     * survey_good_angle, or survey_angle_margin more than the most straight-down view of any survey (never beyond
     * survey_max_angle). Inside a survey's own area its best view is a few degrees off straight down, so it keeps that
     * area; at its edge, where it sees the face only at a slant, an older survey that sees it clearly straighter down
     * paints it instead (a slanted view lays roof edges onto the ground beside them). Not texrecon options. */
    float survey_good_angle = 10.0f;
    float survey_angle_margin = 5.0f;

    /* PATCH11: in nadir mode (the 2.5D mesh of the orthophoto) a face more than steep_face_angle from level (a wall the
     * mesh made a slope) is painted only from its views within steep_view_margin of its most straight-down view. 0: off.
     * Not texrecon options. */
    float steep_face_angle = 45.0f;
    float steep_view_margin = 3.0f;
};

TEX_NAMESPACE_END

template <> inline
const std::vector<std::string> choice_strings<tex::DataTerm>() {
    return {"area", "gmi"};
}

template <> inline
const std::vector<std::string> choice_strings<tex::SmoothnessTerm>() {
    return {"potts"};
}

template <> inline
const std::vector<std::string> choice_strings<tex::OutlierRemoval>() {
    return {"none", "gauss_damping", "gauss_clamping"};
}

template <> inline
const std::vector<std::string> choice_strings<tex::ToneMapping>() {
    return {"none", "gamma"};
}

#endif /* TEX_SETTINGS_HEADER */
