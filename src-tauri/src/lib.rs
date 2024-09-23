use std::{
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::Result;
use app::App;
use tauri::Manager;

mod ann;
mod app;
mod docs;
mod embed;
mod gen;
mod layout;
mod stella;
mod store;
mod utils;

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
async fn index(app: tauri::State<'_, App>, dir: &str) -> Result<(), String> {
    let selected = PathBuf::from_str(dir).map_err(|e| e.to_string())?;
    if !selected.is_dir() {
        return Err(format!("Selected `{dir}` is not a valid directory"));
    }

    app.send(app::Event::Index(selected))
        .await
        .map_err(|e| e.to_string())?;

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // init our app
            let state = App::new(app.path().app_data_dir()?.as_path(), Path::new("../models"))?;
            app.manage(state);

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![index])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
