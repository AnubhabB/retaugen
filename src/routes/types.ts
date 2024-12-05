export interface SearchConfig {
    with_bm25: boolean,
    max_result: number,
    ann_cutoff: number,
    n_sub_qry: number,
    k_adjacent: number,
    relevance_cutoff: number,
}

export interface StatusData {
    head: string,
    body: string,
    hint?: string,
    time_s?: number
}

export interface SearchResult {
    qry: string,
    files: [number, any][],
    evidence: string[],
    answer: string,
    cfg: SearchConfig
}

export interface IndexStat {
    msg: string,
    progress: number,
    pages: number,
    files: number
}