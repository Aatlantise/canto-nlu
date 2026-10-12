import json, random, re
from argparse import ArgumentParser

GRAMMAR = {
    "NP": [
        ("NP_pronoun", "shared", ["pronoun"]),
        ("NP_noun", "shared", ["noun_person"]),
        ("NP_demonstrative", "shared", ["demonstrative", "個", "noun_person"]),
        ("NP_counted_person", "shared", ["number", "個", "noun_person"]),
        ("5.1_NP_plural_mando", "mando", ["noun_person", "們"]),
        ("6.2_NP_plural_dem", "shared", ["demonstrative_plural", "noun_person"]),
    ],
    "NP_thing": [
        ("NPt_classifier", "shared", ["classifier_noun"]),
        ("NPt_counted", "shared", ["number", "classifier_noun"]),
        ("NPt_demonstrative", "shared", ["demonstrative", "classifier_noun"]),
        ("6.2_NPt_plural_dem", "shared", ["demonstrative_plural", "noun_thing"]),
        ("6.2_NPt_di_canto", "canto", ["啲", "noun_thing"]),
    ],
    "NP_thing_subj": [
        ("NPs_demonstrative", "shared", ["demonstrative", "classifier_noun"]),
        ("6.2_NPs_plural_dem", "shared", ["demonstrative_plural", "noun_thing"]),
        ("6.2.1_NPs_bare_classifier", "canto", ["classifier_noun"]),
        ("6.3_NPs_poss_cl_canto", "canto", ["pronoun", "classifier_noun"]),
        ("6.3.1_NPs_possessive", "shared", ["pronoun", "possessive", "noun_thing"]),
    ],
    "NP_thing_def": [
        ("NPd_demonstrative", "shared", ["demonstrative", "classifier_noun"]),
        ("NPd_plural_dem", "shared", ["demonstrative_plural", "noun_thing"]),
        ("6.2.1_NPd_bare_classifier_canto", "canto", ["classifier_noun"]),
        ("NPd_bare_noun_mando", "mando", ["noun_thing"]),
    ],
    "FURN": [
        ("furn_bare", "shared", ["furniture"]),
        ("furn_demonstrative", "shared", ["demonstrative", "furniture_cl"]),
        ("furn_bare_cl_canto", "canto", ["furniture_cl"]),
    ],
    "DURP": [("DURP", "shared", ["number", "time_unit"])],
    "VP": [
        ("VP_intransitive", "shared", ["verb_intransitive", "perfective"]),
        ("VP_duration_intransitive", "shared", ["verb_intransitive", "perfective", "DURP"]),
        ("VP_habitual", "shared", ["adverb_habitual", "verb_transitive", "noun_thing"]),
        ("VP_perfective", "shared", ["verb_transitive", "perfective", "NP_thing"]),
        ("VP_past", "shared", ["adverb_past", "verb_transitive", "perfective", "NP_thing"]),
        ("VP_experiential", "shared", ["verb_transitive", "過", "noun_thing"]),
        ("VP_particle", "shared", ["verb_transitive", "verbal_particle", "noun_thing"]),
        ("VP_duration", "shared", ["verb_durative", "perfective", "DURP", "noun_thing"]),
        ("VP_neg", "shared", ["negator", "verb_transitive", "noun_thing"]),
        ("VP_neg_exp", "shared", ["negator_exp", "verb_transitive", "過", "noun_thing"]),
        ("10.3.3.3_first", "shared", ["先", "verb_intransitive"]),
        ("14.2.1_more_less_canto", "canto", ["verb_transitive", "adjective_more_or_less", "啲", "noun_thing"]),
        ("14.2.1_more_less_mando", "mando", ["adjective_more_or_less", "verb_transitive", "一點", "noun_thing"]),
        ("18.3.6_first_canto", "canto", ["verb_intransitive", "先"]),
    ],
    "VP_bare": [
        ("bare_i", "shared", ["verb_intransitive"]),
        ("bare_t", "shared", ["verb_transitive", "noun_thing"]),
    ],
    "VP_errand": [
        ("errand_i", "shared", ["verb_intransitive"]),
        ("errand_t", "shared", ["verb_errand", "noun_thing"]),
    ],
    "NEG_OPT": [
        ("neg_none", "shared", []),
        ("neg", "shared", ["negator"]),
    ],
    "ASP_OPT": [
        ("asp_none", "shared", []),
        ("asp_pfv", "shared", ["perfective"]),
        ("asp_exp", "shared", ["過"]),
    ],
    "ADV_PAST_OPT": [
        ("adv_past_none", "shared", []),
        ("adv_past", "shared", ["adverb_past"]),
    ],
    "ADV_PRESENT_OPT": [
        ("adv_present_none", "shared", []),
        ("adv_present", "shared", ["adverb_present"]),
    ],
    "ADV_PRESENT_OR_PAST_OPT": [
        ("adv_none", "shared", []),
        ("adv_present", "shared", ["adverb_present"]),
        ("adv_past", "shared", ["adverb_past"]),
    ],
    "DEGREE_OPT": [
        ("degree_none", "shared", []),
        ("degree", "shared", ["degree"]),
    ],
    "POSS_OPT": [
        ("poss_none", "shared", []),
        ("poss", "shared", ["possessive"]),
    ],
    "TOGETHER_OPT": [
        ("together_none", "shared", []),
        ("together", "shared", ["together"]),
    ],
    "Q_FINAL_OPT": [
        ("q_none", "shared", []),
        ("q_final", "shared", ["q_final"]),
    ],
    "RESULT_POT": [
        ("pot_pos", "shared", ["得"]),
        ("pot_neg", "shared", ["negator"]),
    ],
    "DELIM_LEAD": [
        ("lead_want", "shared", ["NP", "想"]),
        ("lead_go", "shared", ["NP", "去"]),
        ("lead_can", "shared", ["NP", "可以"]),
        ("lead_together", "shared", ["NP", "together"]),
        ("lead_let", "shared", ["let_marker", "我"]),
        ("lead_please", "shared", ["請", "你"]),
    ],
    "S": [
        ("4.1", "shared", ["NP", "VP"]),
        ("4.1.3_right_dislocation_canto", "canto", ["degree_canto", "adjective", "sfp_exclaim", "NP_thing_def"]),

        ("6.3.1_poss_pred", "shared", ["NP_thing_subj", "copula", "pronoun", "possessive"]),

        ("7.1.1_location", "shared", ["NP", "locative", "place"]),
        ("7.1.1_location_vp", "shared", ["NP", "locative", "place", "verb_transitive", "noun_thing"]),
        ("7.1.1_here_there", "shared", ["NP", "NEG_OPT", "locative", "here_there"]),
        ("7.1.1_localizer", "shared", ["NP_thing_subj", "locative", "FURN", "localizer"]),

        ("8.1.1_copula", "shared", ["NP", "NEG_OPT", "copula", "noun_person"]),
        ("8.2.2_give_canto", "canto", ["NP", "verb_give", "NP_thing", "NP"]),
        ("8.2.2_give_mando", "mando", ["NP", "verb_give", "NP", "NP_thing"]),
        ("8.3.1_dir", "shared", ["NP", "verb_manner", "directional", "deictic"]),
        ("8.3.1_dir_past_canto", "shared", ["NP", "ADV_PAST_OPT", "verb_manner", "perfective", "directional", "deictic"]),
        ("8.3.1_dir_past_mando", "mando", ["NP", "ADV_PAST_OPT", "verb_manner", "directional", "deictic", "perfective"]),
        ("8.3.1_dir2_past_canto", "canto", ["NP", "ADV_PAST_OPT", "directional", "perfective", "deictic"]),
        ("8.3.1_dir2_past_mando", "mando", ["NP", "ADV_PAST_OPT", "directional", "deictic", "perfective"]),
        ("8.3.3_jeung_place", "shared", ["NP", "disposal", "NP_thing_def", "verb_put", "locative", "place"]),
        ("8.3.3_jeung_furn", "shared", ["NP", "disposal", "NP_thing_def", "verb_put", "locative", "FURN", "localizer"]),
        ("8.3.3_jeung_mando", "mando", ["NP", "把", "NP_thing_def", "verb_ba", "了"]),
        ("8.4_passive", "shared", ["NP_thing_subj", "ADV_PAST_OPT", "passive_marker", "NP", "verb_transitive", "perfective"]),
        ("8.4_passive_barecl_canto", "canto", ["classifier_noun", "畀", "NP", "verb_transitive", "perfective"]),
        ("8.4_passive_noagent_mando", "mando", ["NP_thing_subj", "ADV_PAST_OPT", "被", "verb_transitive", "perfective"]),
        ("8.5.2_v_away", "shared", ["NP", "verb_away", "perfective", "NP_thing_def"]),

        ("9.2.3_abb", "shared", ["NP_thing_subj", "adjective_abb", "POSS_OPT"]),
        ("9.3_cmp_canto", "canto", ["NP", "adjective", "過", "NP"]),
        ("9.3_cmp_mando", "mando", ["NP", "比", "NP", "adjective"]),
        ("9.3.1_cmp_degree_canto", "canto", ["NP", "adjective", "過", "NP", "cmp_degree"]),
        ("9.3.1_cmp_degree_mando", "mando", ["NP", "比", "NP", "adjective", "cmp_degree"]),
        ("9.3.1_cmp_juhng_canto", "canto", ["NP", "仲", "adjective", "過", "NP"]),
        ("9.3.1_cmp_geng_mando", "mando", ["NP", "比", "NP", "更", "adjective"]),
        ("9.3.1_more_and_more", "shared", ["NP", "more_and_more", "adjective"]),
        ("9.3.2_cmp_equal", "shared", ["NP", "with", "NP", "一樣", "adjective"]),
        ("9.3.3_cmp_neg", "shared", ["NP", "have_not", "NP", "so", "adjective"]),
        ("9.3.4_excessive", "shared", ["NP_thing_subj", "太", "adjective"]),
        ("9.3.4_excessive_canto", "canto", ["NP_thing_subj", "adjective", "過頭"]),

        ("10.2_adv_cmp_canto", "canto", ["NP", "verb_adv", "得", "adj_manner", "過", "NP"]),
        ("10.2_adv_cmp_mando", "mando", ["NP", "verb_adv", "得", "比", "NP", "adj_manner"]),
        ("10.2_adv_cmp_neg", "shared", ["NP", "have_not", "NP", "verb_adv", "得", "so", "adj_manner"]),
        ("10.2_adv_a_bit", "shared", ["pronoun_2nd", "verb_adv", "adj_manner", "a_bit", "suggest_particle"]),

        ("11.2_prog_canto", "canto", ["NP", "ADV_PRESENT_OPT", "verb_transitive", "緊", "noun_thing"]),
        ("11.2_prog_mando", "mando", ["NP", "在", "verb_transitive", "noun_thing"]),
        ("11.2.2_prog_preverbal", "shared", ["NP", "prog_preverbal", "verb_transitive", "noun_thing"]),
        ("11.2.3_cont", "shared", ["NP", "verb_cont", "continuous", "NP_thing"]),
        ("11.2.3_cont_simul", "shared", ["NP", "verb_cont", "continuous", "NP_thing", "verb_motion"]),
        ("11.2.7_delim_redup", "shared", ["DELIM_LEAD", "verb_delim_redup"]),
        ("11.2.7_delim_canto", "canto", ["DELIM_LEAD", "verb_delim", "吓"]),
        ("11.2.7_delim_first", "shared", ["NP", "先", "verb_delim_redup"]),
        ("11.2.7_delim_first_canto", "canto", ["NP", "verb_delim", "吓", "先"]),
        ("11.2.9_habit_gwaan", "shared", ["NP", "ADV_PRESENT_OPT", "verb_hab", "慣", "perfective", "NP_thing_def"]),
        ("11.2.9_habit_hoi_canto", "canto", ["NP", "verb_hab", "開", "NP_thing_def", "嘅"]),
        ("11.2.9_habit_hoi_place_canto", "canto", ["NP", "去開", "嗰間", "place_gaan", "嘅"]),
        ("11.3.1_particle_t", "shared", ["NP", "ADV_PAST_OPT", "verb_particle_t", "perfective", "NP_thing"]),
        ("11.3.1_particle_i", "shared", ["NP", "ADV_PAST_OPT", "verb_particle_i", "perfective"]),
        ("11.3.3_everything_canto", "canto", ["NP", "verb_transitive", "晒", "啲", "noun_thing"]),
        ("11.3.3_everything_mando", "mando", ["NP", "把", "demonstrative_plural", "noun_thing", "都", "verb_transitive", "了"]),
        ("11.3.3_suggestion_canto", "canto", ["pronoun_2nd", "verb_eat", "埋", "demonstrative", "food_cl", "suggest_particle"]),
        ("11.3.3_suggestion_mando", "mando", ["pronoun_2nd", "把", "demonstrative", "food_cl", "也", "verb_eat", "了", "suggest_particle"]),

        ("12.1.2_ability", "shared", ["NP", "NEG_OPT", "can_ability", "skill"]),
        ("12.1.2_ability_q", "shared", ["NP", "can_ability_anota", "skill", "Q_FINAL_OPT"]),
        ("12.1.3_need_not", "shared", ["pronoun_2nd", "need_not", "VP_bare"]),
        ("12.1.3_need_q", "shared", ["NP", "need_anota", "VP_errand", "Q_FINAL_OPT"]),
        ("12.3.1_potential_i", "shared", ["NP", "ADV_PRESENT_OR_PAST_OPT", "NEG_OPT", "可以", "verb_intransitive"]),
        ("12.3.1_potential_i_canto", "canto", ["NP", "ADV_PRESENT_OR_PAST_OPT", "NEG_OPT", "verb_intransitive", "得"]),
        ("12.3.1_potential_t", "shared", ["NP", "ADV_PRESENT_OR_PAST_OPT", "NEG_OPT", "可以", "verb_transitive", "noun_thing"]),
        ("12.3.1_potential_t_canto", "canto", ["NP", "ADV_PRESENT_OR_PAST_OPT", "NEG_OPT", "verb_transitive", "得", "noun_thing"]),
        ("12.3.3_result_pot", "shared", ["NP", "verb_result", "RESULT_POT", "到", "noun_thing"]),

        ("13.3.1_not_yet", "shared", ["NP", "still_not", "VP_bare"]),

        ("14.1.2_every_cl_nodup", "shared", ["每", "classifier_noun", "都", "DEGREE_OPT", "adjective"]),
        ("14.1.2_every_cl_redup_canto", "canto", ["classifier_noun_redup", "都", "DEGREE_OPT", "adjective"]),
        ("14.1.5_any_thing", "shared", ["NP", "what_any", "都", "negator", "verb_transitive"]),
        ("14.1.5_any_person", "shared", ["who", "都", "negator", "VP_bare"]),
        ("14.2.1_one_more_canto", "canto", ["NP", "verb_transitive", "多", "classifier_noun"]),
        ("14.2.1_one_more_mando", "mando", ["NP", "多", "verb_transitive", "一", "classifier_noun"]),
        ("14.3.2_no_one", "shared", ["have_not", "人", "VP_bare"]),
        ("14.3.2_nothing", "shared", ["NP", "have_not", "thing_generic", "verb_transitive"]),

        ("15.2_rel_ge", "shared", ["NP", "verb_transitive", "possessive", "noun_thing", "locative", "place"]),
        ("15.3_rel_classifier_canto", "canto", ["NP", "verb_transitive", "嗰", "classifier_noun", "locative", "place"]),

        ("16.1.1_coordination", "shared", ["NP", "with", "NP", "VP"]),
        ("16.2.1_when_clause", "shared", ["NP", "verb_intransitive", "when_marker", "NP", "VP"]),
        ("16.2.1_before_after", "shared", ["NP", "verb_intransitive", "before_after", "NP", "VP"]),

        ("17_q_have_not_have", "shared", ["NP", "ADV_PRESENT_OPT", "have_not_have", "verb_transitive", "noun_thing"]),
        ("17.1.2_a_not_a", "shared", ["NP", "verb_anota", "noun_thing"]),
        ("17.1.3_yes_or_no", "shared", ["NP", "is_it", "VP"]),
        ("17.1.4_q_yet", "shared", ["NP", "verb_transitive", "perfective", "noun_thing", "yet"]),
        ("17.1.6_tag_confirm", "shared", ["NP", "VP", "tag_confirm", "Q_FINAL_OPT"]),
        ("17.1.6_tag_suggest", "shared", ["pronoun_1pl", "VP_bare", "tag_ok", "Q_FINAL_OPT"]),
        ("17.2_alternative", "shared", ["pronoun_2nd", "VP_bare", "alt_or", "VP_bare", "Q_FINAL_OPT"]),
        ("17.3.1_who", "shared", ["who", "VP"]),
        ("17.3.2_what", "shared", ["NP", "verb_transitive", "ASP_OPT", "what"]),
        ("17.3.3_which", "shared", ["NP", "verb_transitive", "which", "classifier_noun", "Q_FINAL_OPT"]),
        ("17.3.4_where_at", "shared", ["NP", "locative", "where", "VP_bare", "Q_FINAL_OPT"]),
        ("17.3.4_where_go", "shared", ["NP", "去", "where", "Q_FINAL_OPT"]),
        ("17.3.5_when", "shared", ["NP", "when", "VP_bare", "Q_FINAL_OPT"]),
        ("17.3.6_why", "shared", ["NP", "why", "NEG_OPT", "VP_bare", "Q_FINAL_OPT"]),
        ("17.3.7_how", "shared", ["NP", "how", "verb_transitive", "noun_thing", "Q_FINAL_OPT"]),
        ("17.3.8_how_many", "shared", ["NP", "verb_transitive", "perfective", "how_many", "classifier_noun", "Q_FINAL_OPT"]),

        ("18.3.1_sentence_final", "shared", ["NP", "VP", "question_marker"]),
        ("18.3.2_laa_canto", "canto", ["NP", "ADV_PRESENT_OPT", "verb_intransitive", "喇"]),
        ("18.3.2_laa_past_canto", "canto", ["NP", "ADV_PAST_OPT", "verb_intransitive", "perfective", "喇"]),
        ("18.3.3_suggest", "shared", ["pronoun_2nd", "VP_bare", "suggest_particle"]),
        ("18.3.5_restrictive", "shared", ["NP", "only_have", "number_small", "classifier_noun", "restrictive"]),
        ("18.3.6_also_canto", "canto", ["NP", "仲", "verb_transitive", "perfective", "NP_thing", "添"]),
        ("18.3.6_also_mando", "mando", ["NP", "還", "verb_transitive", "perfective", "NP_thing"]),
        ("18.3.6_also_maai_canto", "canto", ["NP", "仲", "verb_transitive", "埋", "NP_thing", "添"]),

        ("19.2_first_person_imp", "shared", ["pronoun_1pl", "TOGETHER_OPT", "VP_bare", "suggest_particle"]),
        ("19.4_neg_imperative", "shared", ["pronoun_2nd", "dont", "VP_bare"]),
    ],
}

LEXICON = {
    "pronoun": [("我", "shared"), ("你", "shared"),
                ("佢", "canto"), ("他", "mando"), ("她", "mando"),
                ("我哋", "canto"), ("我們", "mando"),
                ("你哋", "canto"), ("你們", "mando"),
                ("佢哋", "canto"), ("他們", "mando"), ("她們", "mando")],
    "pronoun_2nd": [("你", "shared"), ("你哋", "canto"), ("你們", "mando")],
    "pronoun_1pl": [("我哋", "canto"), ("我們", "mando"), ("咱們", "mando")],
    "demonstrative": [("呢", "canto"), ("這", "mando"), ("嗰", "canto"), ("那", "mando")],
    "demonstrative_plural": [("呢啲", "canto"), ("這些", "mando"), ("嗰啲", "canto"), ("那些", "mando")],

    "noun_person": [("老師", "shared"), ("學生", "shared"), ("同學", "shared"), ("朋友", "shared"),
                    ("醫生", "shared"), ("司機", "shared"), ("鄰居", "shared"), ("經理", "shared"),
                    ("細路仔", "canto"), ("小孩", "mando"),
                    ("男仔", "canto"), ("男孩", "mando"),
                    ("女仔", "canto"), ("女孩", "mando"),
                    ("後生仔", "canto"), ("年輕人", "shared"),
                    ("差佬", "canto"), ("警察", "shared"), ("公安", "mando"),
                    ("侍應", "canto"), ("服務員", "mando"),
                    ("老細", "canto"), ("老闆", "shared"),
                    ("阿媽", "canto"), ("媽媽", "shared"),
                    ("阿爸", "canto"), ("爸爸", "shared")],
    "noun_thing": [("書", "shared"), ("筆", "shared"), ("報紙", "shared"), ("手機", "shared"), ("電腦", "shared"), ("眼鏡", "shared"), ("紙巾", "shared"),
                   ("嘢", "canto"), ("東西", "mando"),
                   ("背囊", "canto"), ("書包", "shared"),
                   ("遮", "canto"), ("雨傘", "mando"),
                   ("銀包", "canto"), ("錢包", "mando"),
                   ("鎖匙", "canto"), ("鑰匙", "mando"),
                   ("鞋", "shared"), ("衫", "canto"), ("衣服", "mando"),
                   ("車", "shared"), ("單車", "canto"), ("自行車", "mando"), ("的士", "canto"), ("出租車", "mando"),
                   ("蘋果", "shared"), ("牛奶", "shared"), ("蛋糕", "shared"),
                   ("蛋撻", "shared"), ("蛋塔", "mando"),
                   ("雪糕", "canto"), ("冰淇淋", "mando"),
                   ("朱古力", "canto"), ("巧克力", "mando"),
                   ("士多啤梨", "canto"), ("草莓", "mando"),
                   ("牛油", "canto"), ("奶油", "mando"),
                   ("薯仔", "canto"), ("土豆", "mando"),
                   ("粟米", "canto"), ("玉米", "mando")],
    "thing_generic": [("嘢", "canto"), ("東西", "mando")],
    "classifier_noun": [("本書", "shared"), ("支筆", "shared"), ("份報紙", "shared"), ("部手機", "shared"), ("部電腦", "shared"), ("副眼鏡", "shared"), ("包紙巾", "shared"),
                        ("個背囊", "canto"), ("個書包", "shared"),
                        ("把遮", "canto"), ("把雨傘", "mando"),
                        ("個銀包", "canto"), ("個錢包", "mando"),
                        ("條鎖匙", "canto"), ("把鑰匙", "mando"),
                        ("對鞋", "canto"), ("雙鞋", "mando"),
                        ("件衫", "canto"), ("件衣服", "mando"),
                        ("架車", "canto"), ("輛車", "mando"),
                        ("架單車", "canto"), ("輛自行車", "mando"),
                        ("架的士", "canto"), ("輛出租車", "mando"),
                        ("個蘋果", "shared"), ("杯牛奶", "shared"),
                        ("件蛋糕", "canto"), ("塊蛋糕", "mando"),
                        ("球雪糕", "canto"), ("球冰淇淋", "mando"),
                        ("嚿朱古力", "canto"), ("塊巧克力", "mando"),
                        ("粒士多啤梨", "canto"), ("個草莓", "mando"),
                        ("塊牛油", "canto"), ("塊奶油", "mando"),
                        ("個薯仔", "canto"), ("個土豆", "mando"),
                        ("條粟米", "canto"), ("根玉米", "mando")],
    "classifier_noun_redup": [("本本書", "canto"), ("支支筆", "canto"), ("份份報紙", "canto"), ("部部手機", "canto"), ("部部電腦", "canto"),
                              ("個個背囊", "canto"), ("把把遮", "canto"), ("個個銀包", "canto"), ("件件衫", "canto"), ("架架車", "canto"),
                              ("個個蘋果", "canto"), ("杯杯牛奶", "canto"), ("件件蛋糕", "canto"),
                              ("件件蛋撻", "shared"), ("件件蛋塔", "mando"),
                              ("球球雪糕", "canto"), ("嚿嚿朱古力", "canto"), ("粒粒士多啤梨", "canto"), ("塊塊牛油", "canto")],
    "food_cl": [("個蘋果", "shared"),
                ("件蛋糕", "canto"), ("塊蛋糕", "mando"),
                ("件蛋撻", "shared"), ("件蛋塔", "mando"),
                ("球雪糕", "canto"), ("球冰淇淋", "mando"),
                ("嚿朱古力", "canto"), ("塊巧克力", "mando"),
                ("粒士多啤梨", "canto"), ("個草莓", "mando"),
                ("個薯仔", "canto"), ("個土豆", "mando"),
                ("條粟米", "canto"), ("根玉米", "mando")],
    "furniture": [("床", "shared"), ("枱", "canto"), ("桌子", "mando"), ("櫈", "canto"), ("椅子", "mando"), ("櫃", "canto"), ("櫃子", "mando")],
    "furniture_cl": [("張床", "shared"), ("張枱", "canto"), ("張桌子", "mando"), ("張櫈", "canto"), ("張椅子", "mando"), ("個櫃", "canto"), ("個櫃子", "mando")],

    "place": [("學校", "shared"), ("圖書館", "shared"), ("公司", "shared"), ("公園", "shared"), ("醫院", "shared"), ("銀行", "shared"),
              ("餐廳", "shared"), ("商場", "shared"), ("超市", "shared"), ("便利店", "shared"), ("機場", "shared"),
              ("屋企", "canto"), ("家", "mando"),
              ("戲院", "canto"), ("電影院", "mando"),
              ("寫字樓", "canto"), ("辦公樓", "mando"),
              ("差館", "canto"), ("警察局", "mando"),
              ("茶餐廳", "canto"), ("酒樓", "canto"), ("士多", "canto")],
    "place_gaan": [("醫院", "shared"), ("銀行", "shared"), ("餐廳", "shared"), ("超市", "shared"),
                   ("戲院", "canto"), ("茶餐廳", "canto"), ("酒樓", "canto"), ("士多", "canto")],
    "here_there": [("呢度", "canto"), ("呢處", "canto"), ("這裡", "mando"), ("這兒", "mando"),
                   ("嗰度", "canto"), ("那裡", "mando"), ("那兒", "mando")],
    "localizer": [("上面", "shared"), ("下面", "shared"), ("旁邊", "shared"),
                  ("面", "canto"), ("上", "mando"),
                  ("底", "canto"), ("下", "mando"),
                  ("隔籬", "canto"), ("側邊", "canto"), ("度", "canto")],

    "skill": [("打字", "shared"), ("彈琴", "shared"),
              ("煮飯", "shared"), ("做飯", "mando"),
              ("打波", "canto"), ("打籃球", "shared"),
              ("揸車", "canto"), ("開車", "mando"),
              ("踩單車", "canto"), ("騎自行車", "mando"),
              ("游水", "canto"), ("游泳", "mando"),
              ("影相", "canto"), ("拍照", "mando"),
              ("打麻雀", "canto"), ("打麻將", "mando"),
              ("講廣東話", "canto"), ("說廣東話", "mando"),
              ("講法文", "canto"), ("說法語", "mando")],

    "number": [("一", "shared"), ("兩", "shared"), ("三", "shared"), ("四", "shared"), ("五", "shared"), ("十", "shared"),
               ("廿", "canto"), ("二十", "mando"),
               ("幾", "shared"),
               ("好多", "canto"), ("勁多", "canto"), ("很多", "mando"),
               ("好少", "canto"), ("很少", "mando")],
    "number_small": [("一", "shared"), ("兩", "shared"), ("三", "shared")],
    "time_unit": [("分鐘", "shared"),
                  ("個鐘頭", "canto"), ("個小時", "mando"),
                  ("日", "canto"), ("天", "mando"),
                  ("個星期", "shared"), ("個禮拜", "canto"),
                  ("個月", "shared"), ("年", "shared")],

    "adjective": [("高", "shared"), ("長", "shared"), ("大", "shared"), ("細", "canto"), ("小", "mando"),
                  ("新", "shared"), ("舊", "shared"), ("快", "shared"), ("慢", "shared"),
                  ("熱", "shared"), ("暖", "shared"), ("凍", "canto"), ("冷", "mando"),
                  ("舒服", "shared"), ("攰", "canto"), ("累", "mando"),
                  ("平", "canto"), ("便宜", "mando"),
                  ("聰明", "shared"), ("叻", "canto"),
                  ("厲害", "shared"), ("勁", "canto"), ("犀利", "canto"),
                  ("奇怪", "shared"), ("黐線", "canto"), ("癲", "canto"), ("瘋", "mando"),
                  ("靚", "canto"), ("漂亮", "mando"),
                  ("好睇", "canto"), ("好看", "mando"),
                  ("得意", "canto"), ("可愛", "mando")],
    "adjective_abb": [("綠油油", "shared"), ("亂糟糟", "shared"), ("軟綿綿", "shared"), ("香噴噴", "shared"),
                      ("靜英英", "canto"), ("靜悄悄", "mando"),
                      ("黑萌萌", "canto"), ("黑漆漆", "mando"),
                      ("凍冰冰", "canto"), ("冷冰冰", "mando"),
                      ("白雪雪", "canto"), ("白花花", "mando"),
                      ("肥嘟嘟", "canto"), ("圓轆轆", "canto")],
    "adjective_more_or_less": [("多", "shared"), ("少", "shared")],
    "adj_manner": [("快", "shared"), ("慢", "shared"), ("好", "shared")],
    "degree": [("太", "shared"), ("非常", "shared"), ("特別", "shared"), ("超級無敵", "shared"),
               ("好", "canto"), ("很", "mando"),
               ("好鬼", "canto"),
               ("幾", "canto"), ("挺", "mando"),
               ("幾咁", "canto"),
               ("有啲", "canto"), ("有點", "mando")],
    "degree_canto": [("好", "canto"), ("幾", "canto"), ("好鬼", "canto")],
    "cmp_degree": [("十倍", "shared"),
                   ("啲", "canto"), ("一啲", "canto"), ("少少", "canto"), ("一點", "mando"), ("一些", "mando"),
                   ("好多", "canto"), ("很多", "mando"), ("得多", "mando")],
    "more_and_more": [("越嚟越", "canto"), ("越來越", "mando")],
    "so": [("咁", "canto"), ("這麼", "mando"), ("那麼", "mando")],
    "a_bit": [("啲", "canto"), ("一啲", "canto"), ("一點", "mando"), ("點", "mando")],

    "adverb_habitual": [("經常", "shared"),
                        ("成日", "canto"), ("係咁", "canto"), ("時常", "mando"), ("總是", "mando"), ("天天", "mando"),
                        ("間中", "canto"), ("耐唔耐", "canto"), ("偶爾", "mando")],
    "adverb_past": [("已經", "shared"),
                    ("頭先", "canto"), ("剛才", "mando"),
                    ("啱啱", "canto"), ("剛剛", "mando"),
                    ("咁啱", "canto"), ("剛好", "mando"),
                    ("琴日", "canto"), ("尋日", "canto"), ("昨天", "mando"),
                    ("前日", "canto"), ("前天", "mando"),
                    ("上個星期", "shared"), ("上個禮拜", "canto"),
                    ("舊年", "canto"), ("去年", "mando")],
    "adverb_present": [("而家", "canto"), ("宜家", "canto"), ("現在", "mando")],
    "together": [("一齊", "canto"), ("一起", "mando")],

    "verb_transitive": [("買", "shared"), ("賣", "shared"), ("借", "shared"), ("帶", "shared"), ("要", "shared"), ("送", "shared"), ("收", "shared"),
                        ("拎", "canto"), ("攞", "canto"), ("拿", "mando"),
                        ("揾", "canto"), ("找", "mando"),
                        ("執", "canto"), ("撿", "mando"),
                        ("掟", "canto"), ("扔", "mando")],
    "verb_intransitive": [("走", "shared"), ("去", "shared"), ("離開", "shared"), ("坐", "shared"), ("等", "shared"), ("笑", "shared"), ("玩", "shared"), ("休息", "shared"),
                          ("嚟", "canto"), ("來", "mando"),
                          ("企", "canto"), ("站", "mando"),
                          ("喊", "canto"), ("哭", "mando"),
                          ("瞓", "canto"), ("睡", "mando"),
                          ("吹水", "canto"), ("聊天", "mando"),
                          ("沖涼", "canto"), ("洗澡", "mando"),
                          ("行街", "canto"), ("逛街", "mando"),
                          ("返工", "canto"), ("上班", "mando"),
                          ("收工", "canto"), ("下班", "mando"),
                          ("返屋企", "canto"), ("回家", "mando")],
    "verb_durative": [("賣", "shared"), ("帶", "shared"), ("揾", "canto"), ("找", "mando")],
    "verb_result": [("買", "shared"), ("借", "shared"), ("揾", "canto"), ("找", "mando")],
    "verb_errand": [("買", "shared"), ("賣", "shared"), ("借", "shared"), ("帶", "shared"), ("送", "shared"), ("收", "shared"),
                    ("拎", "canto"), ("攞", "canto"), ("拿", "mando"),
                    ("揾", "canto"), ("找", "mando")],
    "verb_hab": [("用", "shared"), ("買", "shared")],
    "verb_eat": [("食", "canto"), ("吃", "mando")],
    "verb_adv": [("寫", "shared"), ("做", "shared"), ("唱", "shared"), ("跑", "shared"),
                 ("講", "canto"), ("說", "mando"),
                 ("食", "canto"), ("吃", "mando"),
                 ("行", "canto")],
    "verb_put": [("放", "shared"), ("擺", "shared"), ("留", "shared")],
    "verb_away": [("帶走", "shared"), ("借走", "shared"), ("拎走", "canto"), ("攞走", "canto"), ("拿走", "mando")],
    "verb_ba": [("帶走", "shared"), ("借走", "shared"), ("拿走", "mando"), ("賣掉", "mando")],
    "verb_give": [("畀", "canto"), ("俾", "canto"), ("給", "mando")],
    "verb_cont": [("帶", "shared"), ("抱", "shared"), ("拎", "canto"), ("揸", "canto"), ("拿", "mando")],
    "verb_motion": [("走", "shared"), ("跑", "shared"), ("行", "canto"),
                    ("嚟", "canto"), ("來", "mando"),
                    ("返屋企", "canto"), ("回家", "mando")],
    "verb_manner": [("跳", "shared"), ("爬", "shared"), ("行", "canto"), ("跑", "mando")],
    "verb_delim": [("試", "shared"), ("坐", "shared"),
                   ("睇", "canto"), ("看", "mando"),
                   ("諗", "canto"), ("想", "mando"),
                   ("行", "canto")],
    "verb_delim_redup": [("試一試", "shared"), ("坐一坐", "shared"),
                         ("睇一睇", "canto"), ("看一看", "mando"),
                         ("諗一諗", "canto"), ("想一想", "mando"),
                         ("行一行", "canto"), ("走一走", "mando")],
    "verb_particle_t": [("放低", "canto"), ("放下", "mando"), ("拎起", "canto"), ("拿起", "mando")],
    "verb_particle_i": [("坐低", "canto"), ("坐下", "mando"), ("瞓低", "canto"), ("躺下", "mando")],
    "verb_anota": [("買唔買", "canto"), ("買不買", "mando"),
                   ("借唔借", "canto"), ("借不借", "mando"),
                   ("要唔要", "canto"), ("要不要", "mando")],
    "directional": [("出", "shared"), ("上", "shared"), ("過", "shared"),
                    ("入", "canto"), ("進", "mando"),
                    ("落", "canto"), ("下", "mando"),
                    ("返", "canto"), ("回", "mando")],
    "deictic": [("去", "shared"), ("嚟", "canto"), ("來", "mando")],

    "perfective": [("咗", "canto"), ("了", "mando")],
    "verbal_particle": [("完", "shared"), ("晒", "canto")],
    "continuous": [("住", "canto"), ("著", "mando")],
    "prog_preverbal": [("喺度", "canto"), ("正在", "mando")],

    "can_ability": [("識", "canto"), ("會", "mando")],
    "can_ability_anota": [("識唔識", "canto"), ("會不會", "mando")],
    "need_not": [("唔使", "canto"), ("不用", "mando"), ("不必", "mando")],
    "need_anota": [("使唔使", "canto"), ("需不需要", "mando"), ("用不用", "mando")],
    "have_not_have": [("有冇", "canto"), ("有沒有", "mando")],

    "negator": [("唔", "canto"), ("不", "mando")],
    "negator_exp": [("冇", "canto"), ("沒有", "mando"), ("未", "canto"), ("還沒", "mando")],
    "have_not": [("冇", "canto"), ("沒有", "mando")],
    "still_not": [("仲未", "canto"), ("還沒", "mando"), ("還沒有", "mando")],
    "dont": [("唔好", "canto"), ("咪", "canto"), ("不要", "mando"), ("別", "mando")],

    "copula": [("係", "canto"), ("是", "mando")],
    "possessive": [("嘅", "canto"), ("既", "canto"), ("的", "mando")],
    "locative": [("喺", "canto"), ("在", "mando")],
    "with": [("同", "canto"), ("同埋", "canto"), ("和", "mando"), ("跟", "mando")],
    "disposal": [("將", "shared"), ("把", "mando")],
    "passive_marker": [("畀", "canto"), ("俾", "canto"), ("被", "mando")],
    "let_marker": [("畀", "canto"), ("俾", "canto"), ("讓", "mando")],
    "when_marker": [("嘅時候", "canto"), ("的時候", "mando"), ("時", "mando")],
    "before_after": [("之前", "shared"), ("以前", "shared"), ("之後", "shared"), ("以後", "shared")],
    "only_have": [("只有", "shared"), ("得", "canto"), ("淨係有", "canto")],

    "who": [("邊個", "canto"), ("邊位", "canto"), ("誰", "mando"), ("哪位", "mando")],
    "what": [("乜嘢", "canto"), ("咩嘢", "canto"), ("咩", "canto"), ("什麼", "mando")],
    "what_any": [("乜嘢", "canto"), ("咩嘢", "canto"), ("咩", "canto"), ("什麼", "mando"), ("甚麼", "mando")],
    "which": [("邊", "canto"), ("哪", "mando")],
    "where": [("邊度", "canto"), ("邊", "canto"), ("哪裡", "mando"), ("哪兒", "mando")],
    "when": [("幾時", "canto"), ("什麼時候", "mando")],
    "why": [("點解", "canto"), ("做咩", "canto"), ("為什麼", "mando"), ("為甚麼", "mando")],
    "how": [("點樣", "canto"), ("點", "canto"), ("怎樣", "mando"), ("怎麼", "mando")],
    "how_many": [("幾", "shared"), ("幾多", "canto"), ("多少", "mando")],

    "question_marker": [("咩", "canto"), ("嗎", "mando")],
    "q_final": [("啊", "shared"), ("呀", "canto"), ("呢", "mando")],
    "yet": [("未", "canto"), ("沒有", "mando")],
    "is_it": [("係咪", "canto"), ("是不是", "mando")],
    "tag_confirm": [("係咪", "canto"), ("係唔係", "canto"), ("是不是", "mando"), ("對不對", "mando")],
    "tag_ok": [("好唔好", "canto"), ("得唔得", "canto"), ("好不好", "mando"), ("行不行", "mando")],
    "alt_or": [("定", "canto"), ("定係", "canto"), ("還是", "mando")],
    "sfp_exclaim": [("喎", "canto"), ("呀", "canto"), ("㗎", "canto")],
    "suggest_particle": [("啦", "canto"), ("吧", "mando")],
    "restrictive": [("咋", "canto"), ("啫", "canto"), ("㗎咋", "canto"), ("而已", "mando"), ("罷了", "mando")],
}


HOLDOUT_RULES = {
    "4.1.3_right_dislocation_canto", "7.1.1_localizer",
    "8.2.2_give_canto", "8.2.2_give_mando", "8.3.3_jeung_place", "8.3.3_jeung_furn",
    "9.2.3_abb", "9.3.1_cmp_juhng_canto", "9.3.1_cmp_geng_mando", "10.2_adv_cmp_neg",
    "11.2.3_cont", "11.2.3_cont_simul", "11.3.3_suggestion_canto", "11.3.3_suggestion_mando",
    "12.1.2_ability", "12.1.2_ability_q", "13.3.1_not_yet",
    "14.2.1_one_more_canto", "14.2.1_one_more_mando", "14.3.2_no_one", "14.3.2_nothing",
    "15.2_rel_ge", "15.3_rel_classifier_canto", "16.2.1_when_clause",
    "17.2_alternative", "17.3.6_why", "18.3.5_restrictive", "19.2_first_person_imp",
}
HOLDOUT_LEX = {
    "醫生", "侍應", "服務員", "後生仔", "公安",
    "電腦", "銀包", "錢包", "鎖匙", "鑰匙", "雪糕", "冰淇淋",
    "圖書館", "機場", "戲院", "電影院", "士多",
    "影相", "拍照", "廿", "二十",
    "平", "便宜", "聰明", "靚", "漂亮", "特別",
    "前日", "前天", "舊年", "去年", "宜家",
    "送", "揾", "找", "執", "撿", "瞓", "睡", "休息", "行街", "逛街", "收工", "下班",
    "俾", "既",
}

MODES = {"canto": {"canto", "shared"}, "mando": {"mando", "shared"},
         "other": {"canto", "mando", "shared"}}
MODE_TO_ID = {"canto": 0, "mando": 1, "other": 2}
TAGS = {"canto", "mando", "shared"}


class DeadEnd(Exception):
    pass


def is_held_word(w: str) -> bool:
    return any(h in w for h in HOLDOUT_LEX)


def expand(sym: str, allowed: set[str], trace: dict, exclude: bool = False) -> str:
    '''recursively expand a symbol. we keep track of the rules and tags used in the trace'''
    if sym in GRAMMAR:
        options = [r for r in GRAMMAR[sym] if r[1] in allowed and not (exclude and r[0] in HOLDOUT_RULES)]
        if not options:
            raise DeadEnd
        name, tag, rhs = random.choice(options)
        trace["rules"].append(name)
        trace["tags"].add(tag)
        if name in HOLDOUT_RULES:
            trace["held"].add("rule")
        return "".join(expand(s, allowed, trace, exclude) for s in rhs)
    if sym in LEXICON:
        options = [w for w in LEXICON[sym] if w[1] in allowed and not (exclude and is_held_word(w[0]))]
        if not options:
            raise DeadEnd
        word, tag = random.choice(options)
        trace["tags"].add(tag)
        if is_held_word(word):
            trace["held"].add("lex")
        return word
    return sym


def label(tags: set[str]) -> str:
    if {"canto", "mando"} <= tags:
        return "other"
    if "canto" in tags:
        return "canto"
    if "mando" in tags:
        return "mando"
    return "other"


def sample(allowed: set[str], start: str, exclude: bool = False) -> dict:
    '''sample a sentence from the grammar'''
    while True:
        trace = {"rules": [], "tags": set(), "held": set()}
        try:
            text = expand(start, allowed, trace, exclude)
        except DeadEnd:
            continue
        return {"sentence": text, "label": MODE_TO_ID[label(trace["tags"])],
                "holdout": "+".join(sorted(trace["held"]))}


def dataset(n: int, start: str, test: bool = False, train_seen: set[str] = None, seed: int = 0, max_tries: int = 1000000) -> dict[str, list[dict]]:
    random.seed(seed)
    seen, out = set(), {k: [] for k in MODES}
    for mode in MODES:
        allowed = MODES[mode]
        count = 0
        for _ in range(max_tries):
            ex = sample(allowed, start, exclude=not test)
            if test and (not ex["holdout"] or ex["holdout"] != "lex+rule" or ex["sentence"] in train_seen):
                continue
            if not test:
                del ex["holdout"]
            if ex["label"] == MODE_TO_ID[mode] and ex["sentence"] not in seen:
                seen.add(ex["sentence"])
                out[mode].append(ex)
                count += 1
                if count == n:
                    break
        if count < n:
            print(f"only found {count} unique '{mode}' sentences")
    return out


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_train", type=int, default=10000, help="train sentences per mode")
    parser.add_argument("--n_test", type=int, default=2000, help="test sentences per mode")
    parser.add_argument("--start", type=str, default="S", help="start symbol (default is S)")
    args = parser.parse_args()
    print(f"grammar: {sum(len(rules) for rules in GRAMMAR.values())} rules")
    print(f"lexicon: {sum(len(entries) for entries in LEXICON.values())} lexical items")
    print(f'held-out rules: {len(HOLDOUT_RULES)}')
    print(f'held-out lexicon: {len(HOLDOUT_LEX)}')
    train_seen = set()
    for split, n in [("train", args.n_train), ("test", args.n_test)]:
        data = dataset(n, args.start, test=(split == "test"), train_seen=train_seen)
        with open(f"lid_{split}.jsonl", "w", encoding="utf-8") as f:
            for mode in MODES:
                f.writelines(json.dumps(ex, ensure_ascii=False) + "\n" for ex in data[mode])
                print(f"wrote {len(data[mode])} {mode} {split} sentences")
                if split == "train":
                    train_seen.update(ex["sentence"] for ex in data[mode])
