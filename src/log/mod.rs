#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {{
        print!("\x1b[36mDEBUG\x1b[0m: {}: ", self::log_info::NAME);
        println!($($arg)*);
    }};
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {{
        print!("\x1b[31mERROR\x1b[0m: {}: ", self::log_info::NAME);
        println!($($arg)*);
    }};
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {{
        print!("\x1b[32mINFO\x1b[0m: {}: ", self::log_info::NAME);
        println!($($arg)*);
    }};
}

#[macro_export]
macro_rules! define_log_info {
    ($name:literal) => {
        mod log_info {
            #[allow(dead_code)]
            pub const NAME: &'static str = $name;
        }
    };
}
