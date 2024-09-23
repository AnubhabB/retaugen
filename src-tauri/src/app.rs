use std::{fs::create_dir_all, path::{Path, PathBuf}, sync::Arc};

use anyhow::Result;
use tauri::async_runtime::{self, channel, Receiver, RwLock, Sender};

use crate::{docs::files_to_text, store::Store};

pub enum Event {
    Search(String),
    Index(PathBuf)
}

#[derive(Clone)]
pub struct App {
    dir: PathBuf,
    send: Sender<Event>,
    store: Arc<RwLock<Store>>
}

impl App {
    /// Create a new instance of the app
    pub fn new(appdir: &Path) -> Result<Self> {
        println!("{appdir:?}");
        let storage_dir = appdir.join("store");
        if !storage_dir.is_dir() {
            create_dir_all(&storage_dir)?;
        }

        let (send, recv) = channel(100);
        let store = Arc::new(
            RwLock::new(Store::load_from_file(storage_dir.as_path(), None, None)?)
        );

        let app = Self {
            dir: storage_dir.to_path_buf(),
            send,
            store
        };

        let arced = Arc::new(app.clone());
        async_runtime::spawn(async move {
            Self::listen(arced, recv).await;
        });

        Ok(app)
    }

    pub async fn send(&self, e: Event) -> Result<()> {
        self.send.send(e).await?;
        Ok(())
    }

    async fn listen(app: Arc<Self>, recv: Receiver<Event>) {
        let mut recv = recv;
        while let Some(evt) = recv.recv().await {
            match evt {
                Event::Search(qry) => {
                    println!("Query: {qry}");
                }
                Event::Index(dir) => {
                    if let Err(e) = app.index(dir.as_path()).await {
                        eprintln!("Error while indexing {dir:?}: {e:?}");
                    }
                }
            }
        }
    }

    // Triggers indexing workflow
    async fn index(&self, path: &Path) -> Result<()> {
        let mut to_index = vec![];
        // Create list of files
        path.read_dir()?.filter_map(|f| {
            if let Ok(p) = f {
                if p.metadata()
                .map_or( false,|f| {
                    f.is_file()
                }) {
                    Some(p)
                } else {
                    None
                }
            } else {
                None
            }
        }).for_each(|f| {
            let path = f.path();
            if let Some(ext) = path.extension() {
                if ext == "txt" || ext == "pdf" {
                    to_index.push(path);
                }
            }
        });

        let data = files_to_text(&to_index[..])?.concat();

        let writer = self.store.write().await;
        // writer.
        Ok(())
    }
}