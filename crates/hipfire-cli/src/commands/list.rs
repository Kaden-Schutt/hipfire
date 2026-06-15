use crate::model::{list_local_models, model_display_name};

pub fn run() {
    let models = list_local_models();
    if models.is_empty() {
        println!("No models found in ~/.hipfire/models/");
    } else {
        for p in &models {
            println!("{}", model_display_name(p));
        }
    }
}
