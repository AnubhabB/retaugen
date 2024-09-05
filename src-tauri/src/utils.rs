use anyhow::{anyhow, Result};
use candle_core::{utils::{cuda_is_available, metal_is_available}, Device, Tensor};

pub fn select_device() -> Result<Device> {
    if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

pub fn masked_fill(lhs: &Tensor, rhs: &Tensor, value: f32) -> candle_core::Result<Tensor> {
    rhs.where_cond(
        lhs,
        &Tensor::new(value, lhs.device())?.broadcast_as(rhs.shape().dims())?,
    )
}