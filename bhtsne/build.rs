// use std::env;
// use std::path::PathBuf;

fn main() {
    // let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .warnings(false)
        .static_flag(true)
        .pic(true)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("/std:c++17")
        .flag_if_supported("-ffast-math")
        .flag_if_supported("/fp:fast")
        .define("BUILDING_TSNE_DLL", None)
        .include("contrib/tsne/tsne/bh_sne_src")
        .file("contrib/tsne/tsne/bh_sne_src/sptree.cpp")
        .file("contrib/tsne/tsne/bh_sne_src/tsne.cpp");

    if cfg!(target_feature = "sse3") {
        build.flag("-msse3");
    }
    if cfg!(target_feature = "ssse3") {
        build.flag("-mssse3");
    }
    if cfg!(target_feature = "sse4.1") {
        build.flag("-msse4.1");
    }
    if cfg!(target_feature = "sse4.2") {
        build.flag("-msse4.2");
    }
    if cfg!(target_feature = "popcnt") {
        build.flag("-mpopcnt");
    }
    if cfg!(target_feature = "cmpxchg16b") && !cfg!(target_os = "macos") {
        // Apple clang version 14.0.0 does not support -mcmpxchg16b.
        // Fix the error: cargo:warning=clang: error: unknown argument: '-mcmpxchg16b'
        if build.get_compiler().is_like_gnu() {
            build.flag("-mcx16");
        } else {
            build.flag("-mcmpxchg16b");
        }
    }
    if cfg!(target_feature = "fma4") {
        build.flag("-mfma4");
    }
    if cfg!(target_feature = "avx") {
        build.flag("-mavx");
    }
    if cfg!(target_feature = "bmi1") || cfg!(target_feature = "bmi2") {
        build.flag("-mbmi");
    }
    if cfg!(target_feature = "lzcnt") {
        build.flag("-mlzcnt");
    }
    if cfg!(target_feature = "movbe") {
        build.flag("-mmovbe");
    }

    build.try_compile("bhtsne.a").expect("could not compile bhtsne.a");

    /*
    bindgen::Builder::default()
        .header("wrapper.h")
        .header("contrib/tsne/tsne/bh_sne_src/tsne.h")
        .generate_comments(false)
        .generate()
        .expect("could not generate bindings")
        .write_to_file(out.join("bindings.rs"))
        .expect("could not write bindings.rs");
    */

    println!("cargo:rerun-if-changed=contrib/tsne/tsne/bh_sne_src/sptree.cpp");
    println!("cargo:rerun-if-changed=contrib/tsne/tsne/bh_sne_src/sptree.h");
    println!("cargo:rerun-if-changed=contrib/tsne/tsne/bh_sne_src/tsne.cpp");
    println!("cargo:rerun-if-changed=contrib/tsne/tsne/bh_sne_src/tsne.h");
    println!("cargo:rerun-if-changed=contrib/tsne/tsne/bh_sne_src/vptree.h");
}
