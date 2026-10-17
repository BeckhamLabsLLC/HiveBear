//! The base override is process-global, so it gets its own test binary.

use hivebear_core::{AppPaths, Config};

#[test]
fn config_follows_base_override() {
    let base = std::env::temp_dir().join(format!("hivebear-override-{}", std::process::id()));
    assert!(AppPaths::set_base_override(base.clone()));
    assert!(!AppPaths::set_base_override(base.join("ignored")));

    let paths = AppPaths::new();
    assert_eq!(paths.config_file, base.join("config").join("config.toml"));
    assert_eq!(paths.db_file, base.join("data").join("hivebear.db"));
    assert_eq!(
        Config::default().models_dir,
        base.join("data").join("models")
    );

    let mut config = Config::load();
    config.mesh.enabled = true;
    config.mesh.auto_join = true;
    config.save().expect("save under the override");
    assert!(paths.config_file.exists());
    let reloaded = Config::load();
    assert!(reloaded.mesh.enabled && reloaded.mesh.auto_join);

    std::fs::remove_dir_all(&base).ok();
}
