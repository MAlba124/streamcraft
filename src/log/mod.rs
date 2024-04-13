#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {{
        print!("DEBUG: {}: ", self::log_info::NAME);
        println!($($arg)*);
    }};
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {{
        print!("ERROR: {}: ", self::log_info::NAME);
        println!($($arg)*);
    }};
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {{
        print!("INFO: {}: ", self::log_info::NAME);
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
