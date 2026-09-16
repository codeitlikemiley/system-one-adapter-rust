use serde_json::json;
use system_one_adapter::schema::{create_llm_output_model, create_raw_output_schema};
use system_one_adapter::types::{convert_question_collection, AnswerMode, Choice, Noul, Question};

const SCHEMA_KEYWORDS: &[&str] = &[
    "title",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
];

#[test]
fn question_ids_can_match_schema_keywords_probabilities() {
    question_ids_can_match_schema_keywords(AnswerMode::Probabilities);
}

#[test]
fn question_ids_can_match_schema_keywords_discrete() {
    question_ids_can_match_schema_keywords(AnswerMode::Discrete);
}

fn question_ids_can_match_schema_keywords(answer_mode: AnswerMode) {
    let questions = convert_question_collection(SCHEMA_KEYWORDS.iter().map(|key| {
        (
            (*key).to_string(),
            Question::from(Noul::new(format!("Evaluate {key}."))),
        )
    }))
    .unwrap();
    let output_model = create_llm_output_model(&questions, answer_mode);
    let schema = create_raw_output_schema(&output_model);
    let answers = &schema["$defs"]["TypeSafeAnswers"];

    assert_eq!(
        schema["properties"]["answers"],
        json!({ "$ref": "#/$defs/TypeSafeAnswers" })
    );
    assert!(answers["description"]
        .as_str()
        .unwrap()
        .contains("Use these property names verbatim"));
    let properties = answers["properties"].as_object().unwrap();
    let required: Vec<&str> = answers["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(properties.len(), questions.len());
    assert_eq!(required.len(), questions.len());
    for key in SCHEMA_KEYWORDS {
        assert!(properties.contains_key(*key));
        assert!(required.contains(key));
        let answer = &properties[*key];
        let instructions = questions.get(key).unwrap().instructions().unwrap();
        assert!(answer["description"]
            .as_str()
            .unwrap()
            .contains(instructions.as_str().unwrap()));
        for keyword in SCHEMA_KEYWORDS {
            assert!(
                !answer.as_object().unwrap().contains_key(*keyword),
                "{keyword} should not be a schema keyword on the answer property"
            );
        }
    }

    let mut payload_answers = serde_json::Map::new();
    for key in SCHEMA_KEYWORDS {
        payload_answers.insert(
            (*key).to_string(),
            if answer_mode == AnswerMode::Discrete {
                json!(true)
            } else {
                json!(0.8)
            },
        );
    }
    let payload = json!({ "answers": payload_answers });
    let decoded = output_model
        .decode(&serde_json::to_string(&payload).unwrap())
        .unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn probability_labels_can_match_schema_keywords() {
    let criteria: indexmap::IndexMap<String, Option<serde_json::Value>> = SCHEMA_KEYWORDS
        .iter()
        .map(|key| {
            (
                (*key).to_string(),
                Some(serde_json::Value::String(format!("The {key} option."))),
            )
        })
        .collect();
    let questions = convert_question_collection([(
        "level".to_string(),
        Question::from(Choice::from_criteria(criteria.clone())),
    )])
    .unwrap();
    let output_model = create_llm_output_model(&questions, AnswerMode::Probabilities);
    let schema = create_raw_output_schema(&output_model);
    let probabilities = &schema["$defs"]["ProbabilityMap0"];

    let properties = probabilities["properties"].as_object().unwrap();
    let required: Vec<&str> = probabilities["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(properties.len(), SCHEMA_KEYWORDS.len());
    assert_eq!(required.len(), SCHEMA_KEYWORDS.len());
    assert!(probabilities.get("title").is_none() || probabilities["title"].is_null());
    // The definition itself must not carry a title keyword.
    assert!(!probabilities.as_object().unwrap().contains_key("title"));
    for key in SCHEMA_KEYWORDS {
        assert!(properties.contains_key(*key));
        assert!(required.contains(key));
        assert_eq!(
            properties[*key]["description"],
            json!(format!("The {key} option."))
        );
        for keyword in SCHEMA_KEYWORDS {
            assert!(!properties[*key].as_object().unwrap().contains_key(*keyword));
        }
    }

    let mut invalid = serde_json::Map::new();
    for key in SCHEMA_KEYWORDS {
        invalid.insert((*key).to_string(), json!(2));
    }
    let payload = json!({ "answers": { "level": invalid } });
    assert!(output_model
        .decode(&serde_json::to_string(&payload).unwrap())
        .is_err());
}
