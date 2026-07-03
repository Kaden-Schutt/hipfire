fn main() {
    match hipfire_xdna::XdnaDevice::open_default() {
        Ok(dev) => {
            println!("path: {}", dev.path());
            match dev.resource_info() {
                Ok(ri) => println!("resource_info: {ri:#?}"),
                Err(e) => println!("resource_info err: {e}"),
            }
            match dev.clocks() {
                Ok(c) => println!("clocks: {c:#?}"),
                Err(e) => println!("clocks err: {e}"),
            }
        }
        Err(e) => println!("open err: {e}"),
    }
}
