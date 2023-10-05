#[allow(unused_extern_crates)]
#[cfg(target_os = "macos")]
extern crate accelerate_src;

#[allow(unused_extern_crates)]
#[cfg(not(target_os = "macos"))]
extern crate intel_mkl_src;
