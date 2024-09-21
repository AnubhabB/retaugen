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

/// The end text of prev would be common with the begining of current
/// E.g.
/// prev: Hello, how are you? Life is good!
/// current: Life is good! The act of creation keeps us buzy!
// The strategy is simple, we'll pick up the midpoint of current and keep checking backwards if prev ends with the text
// This will be pretty efficient because with max token size of 512 and a known overlap factor which is 1/4th of the token size
// this would yield reasonable results
pub fn dedup_text(prev: &str, current: &str) -> Result<String> {
    let cur = current.as_bytes();
    let prv = prev.as_bytes();

    let mut pointer = cur.len() / 2;

    while pointer > 0 {
        if prv.ends_with(&cur[0..pointer]) {
            break;
        }
        pointer -= 1;
    }

    Ok(std::str::from_utf8(&prv[0..prv.len() - pointer])?.to_string())
}

#[cfg(test)]
mod tests {
    use super::dedup_text;

    #[test]
    fn test_dedup_text() {
        let prv = "Futtuh, head of the Palestinian parliament, was sworn in hours after the death of Yasser Arafat on Thursday, and Palestinian Basic Law dictates that he may only serve up to two months before elections are held.\n\nNew leadership could prove to be the key to revitalizing the peace process in the Middle East, as both Israel and the United States had refused to work with Arafat.\n\nThe Haaretz had initially reported that former prime minister Mahmoud Abbas was selected by the Fatah central committee as their candidate for president, but Abbas has denied this, saying, \"the matter is still being discussed.\" There have also been conflicting reports on whether or not jailed Palestinian leader Marwan Barghouti will run.\n\nBarghouti is currently serving five life sentences in Israel for attacks against Israelis. Nonetheless, he remains a popular figure among Palestinians for his role in the Palestinian uprising, and could potentially win the election if he decided to run.";

        let cur = "The Haaretz had initially reported that former prime minister Mahmoud Abbas was selected by the Fatah central committee as their candidate for president, but Abbas has denied this, saying, \"the matter is still being discussed.\" There have also been conflicting reports on whether or not jailed Palestinian leader Marwan Barghouti will run.\n\nBarghouti is currently serving five life sentences in Israel for attacks against Israelis. Nonetheless, he remains a popular figure among Palestinians for his role in the Palestinian uprising, and could potentially win the election if he decided to run.\n\nA win by Barghouti could put Israel in an awkward spot; however an Israeli official said this week that he would not be freed, and a landslide win by Barghouti would signify to them that the Palestinians were not yet ready for peace.## Brazilian delegation returns from Arafat funeral\nPalestineThe delegation representing Brazil at the funeral of Yasser Arafat returned today, November 13, 2004. The chief-minister of Civil House José Dirceu was a member of the delegation. Unfortunately they arrived too late for the funeral and the delegation watched only part of the funeral activities.\n\nPCdoB (Brazilian communist political party) Deputy Jamil Murad, member of the delegation, said there was a \"deep mourning\" feeling. Jamil Murad had visited Yasser Arafat in April 2004, along with nine other Brazilian deputies. According to Jamil Murad: \"Yasser Arafat was a Palestinian leader who became a world projection leader\". He said Arafat had written him a letter thanking the Brazilian people for their support of the Palestinian cause and saying that he, Arafat, considered President Luiz Inácio Lula da Silva a great world leader.## Hearing begins over David Hookes death\nA hearing started today over the death of Australian cricket coach David Hookes. Hookes died after an incident outside a hotel in Melbourne, Australia on the 19th of January.\n\nBouncer Zdravko Micevic, 22, is charged with manslaughter.";

        let p = dedup_text(prv, cur).unwrap();

        assert_eq!(&p, "Futtuh, head of the Palestinian parliament, was sworn in hours after the death of Yasser Arafat on Thursday, and Palestinian Basic Law dictates that he may only serve up to two months before elections are held.\n\nNew leadership could prove to be the key to revitalizing the peace process in the Middle East, as both Israel and the United States had refused to work with Arafat.\n\n");

        let prv = "Hello, this is great! a";
        let cur = "a new dawn, a new light!";

        let p = dedup_text(prv, cur).unwrap();
        assert_eq!(&p, "Hello, this is great! ");
    }
}
