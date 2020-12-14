extern crate csv;
extern crate itertools;
extern crate xgboost;

use array_tool::vec::Uniq;
use lindera::tokenizer::Tokenizer;
use lindera_core::core::viterbi::Mode;
use serde::Deserialize;
use fasttext::FastText;
use fasttext::Args;
use itertools::zip;
use std::io::Write;
use std::fs::File;
use xgboost::{DMatrix, Booster};
use xgboost::parameters::{self, tree, learning::Objective};


#[derive(Debug, Deserialize)]
struct Record {
    text: String,
    label: f32
}


fn main() -> std::io::Result<()> {
    // load train data
    let mut rdr = csv::Reader::from_path("./data/train.csv")?;
    let mut train_sentences = Vec::new();
    let mut train_labels = Vec::new();
    for result in rdr.deserialize() {
        let record: Record = result?;
        train_labels.push(record.label);
        train_sentences.push(record.text);
    }

    // tokenize sentence & save file for fasttext
    let mut tokenizer = Tokenizer::new(Mode::Normal, "./lindera-ipadic-2.7.0-20070801-neologd-20200910");
    let mut file = File::create("./data/train_fasttext.csv")?;
    for (sentence, label) in zip(&train_sentences, &train_labels){
        let row = tokenizer.tokenize(&sentence).iter().map(|x| x.text).collect::<Vec<&str>>().join(" ");
        write!(file, "__label__{}, {}\n", label, row)?;
    }
    file.flush()?;

    // train fasttext & save model
    let mut fasttext_args = Args::default();
    fasttext_args.set_input("./data/train_fasttext.csv");
    let mut model = FastText::default();
    let _ = model.train(&fasttext_args);
    let _ = model.save_model("./data/fasttext.bin");

    // load test data
    let mut rdr = csv::Reader::from_path("./data/test.csv")?;
    let mut test_sentences = Vec::new();
    let mut test_labels = Vec::new();
    for result in rdr.deserialize() {
        let record: Record = result?;
        test_labels.push(record.label);
        test_sentences.push(record.text);
    }

    // vectorize fasttext
    let train_ft_vector_flatten = &train_sentences.iter().map(|x| model.get_sentence_vector(x)).collect::<Vec<Vec<_>>>().into_iter().flatten().collect::<Vec<_>>();
    let test_ft_vector_flatten = &test_sentences.iter().map(|x| model.get_sentence_vector(x)).collect::<Vec<Vec<_>>>().into_iter().flatten().collect::<Vec<_>>();

    // make dataset
    let train_data_size = train_sentences.len();
    let test_data_size = test_sentences.len();
    let mut train_dmat = DMatrix::from_dense(train_ft_vector_flatten, train_data_size).unwrap();
    train_dmat.set_labels(&train_labels).unwrap();
    let mut test_dmat = DMatrix::from_dense(test_ft_vector_flatten, test_data_size).unwrap();
    test_dmat.set_labels(&test_labels).unwrap();

    // train xgboost
    let uniq_label = train_labels.unique().len() as u32;
    let eval_sets = &[(&train_dmat, "train"), (&test_dmat, "test")];
    let learning_params = parameters::learning::LearningTaskParametersBuilder::default().objective(Objective::MultiSoftmax(uniq_label)).build().unwrap();
    let tree_params = tree::TreeBoosterParametersBuilder::default().eta(0.1).max_depth(6).build().unwrap();
    let booster_params = parameters::BoosterParametersBuilder::default().booster_type(parameters::BoosterType::Tree(tree_params)).learning_params(learning_params).build().unwrap();
    let training_params = parameters::TrainingParametersBuilder::default().dtrain(&train_dmat).booster_params(booster_params).boost_rounds(5).evaluation_sets(Some(eval_sets)).build().unwrap();
    let bst = Booster::train(&training_params).unwrap();

    // predict xgboost
    let preds = bst.predict(&test_dmat).unwrap();
    let mut file = File::create("./data/pred.csv")?;
    write!(file, "label, pred\n")?;
    for (label, pred) in zip(&test_labels, &preds){
        write!(file, "{}, {}\n", label, pred)?;
    }
    file.flush()?;

    Ok(())
}
