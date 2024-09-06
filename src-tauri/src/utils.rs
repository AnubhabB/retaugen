use anyhow::Result;
use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    Device,
};

pub fn select_device() -> Result<Device> {
    if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else {
        Ok(Device::Cpu)
    }
}
