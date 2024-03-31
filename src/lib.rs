pub mod element_traits;
pub mod elements;
pub mod pipeline;

#[macro_export]
macro_rules! debug_log {
    ($($arg:expr)*) => {{
        #[cfg(debug_assertions)]
        {
            print!("DEBUG: {}:{} ", file!(), line!());
            println!($($arg)*);
        }
    }};
}
