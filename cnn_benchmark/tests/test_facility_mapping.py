"""Provisional facility map: manuscript composition, explicit provisional cases, nothing silently dropped."""


def test_saratov_primary_denominator_matches_manuscript(facilities):
    sar = facilities[(facilities.region == "Saratov") & facilities.in_primary_denominator]
    assert sar.facility_type.value_counts().to_dict() == {
        "icbm_silo": 20, "warhead_storage": 3, "warhead_rail_to_road_transfer": 1}


def test_candidate_21st_silo_is_kept_and_flagged(facilities):
    c = facilities[facilities.status == "candidate_extra_silo_not_in_manuscript_count"]
    assert len(c) == 1
    c = c.iloc[0]
    assert c.facility_type == "icbm_silo" and c.z16_parent_tiles == "r2_c17"
    assert not c.in_primary_denominator and c.needs_author_confirmation and c.known_target
    assert (facilities.facility_type == "icbm_silo").sum() == 21


def test_non_target_waste_repository_kept(facilities):
    w = facilities[facilities.facility_type == "radioactive_waste_repository"]
    assert len(w) == 1 and not w.known_target.iloc[0] and not w.in_primary_denominator.iloc[0]


def test_iran_single_facilities(facilities):
    for region, tiles in {"Natanz": "r15_c16;r15_c17;r16_c16", "Fordow": "r14_c15;r15_c15;r16_c16"}.items():
        f = facilities[facilities.region == region]
        assert len(f) == 1
        assert f.z16_parent_tiles.iloc[0] == tiles
        assert f.in_primary_denominator.iloc[0] and f.needs_author_confirmation.iloc[0]


def test_no_facility_silently_dropped(reconstruction, membership, parent_table):
    facilities, assignments = reconstruction
    assert set(facilities.facility_id) == set(assignments.facility_id)
    z16_ids = set(facilities.facility_id[facilities.n_z16_parent_tiles > 0])
    assert z16_ids == set(membership.facility_id) == set(facilities.facility_id)   # no z17-only facilities
    keys = set(zip(parent_table.region, parent_table.tile_id))
    assert set(zip(membership.region, membership.tile_id)) <= keys


def test_provisional_cases_require_confirmation(facilities):
    assert facilities[facilities.facility_type == "warhead_storage"].needs_author_confirmation.all()
    reno = facilities[facilities.human_labels.str.contains("under renovation")]
    assert len(reno) == 2 and reno.needs_author_confirmation.all() and reno.in_primary_denominator.all()


def test_each_label_row_assigned_to_one_facility(reconstruction):
    _, a = reconstruction
    assert not a.unit_id.duplicated().any()
