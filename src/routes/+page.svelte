<script lang="ts">
  import { open } from '@tauri-apps/plugin-dialog';
  import { invoke } from "@tauri-apps/api/core";
  import { getCurrentWindow } from '@tauri-apps/api/window';
  import { onMount } from 'svelte';
  import type { IndexStat, SearchConfig, SearchResult, StatusData } from './types';
  import Search from './Search.svelte';

  let searchCfg: SearchConfig = {
    with_bm25: true,
    allow_without_evidence: false,
    max_result: 8,
    ann_cutoff: 0.75,
    n_sub_qry: 3,
    k_adjacent: 1,
    relevance_cutoff: 5.
  };

  let search: string,
    searching: boolean = false;
    
  let indexing: boolean = false,
    idxstatus: IndexStat|undefined;

  let logs: StatusData[] = [];

  let searches: SearchResult[] = [];
  // [{
  //   qry: "Some awesome search query!",
  //   files: [],
  //   evidence: ["Some long evidence text which can make a lot of difference", "More evidence here"],
  //   answer: "Some long gemini style answer",
  //   cfg: searchCfg
  // }];

  const folderPicker = async () => {
    const dir = await open({
      multiple: false,
      directory: true,
    });
    
    if(!dir)
      return;

    
    try {
      invoke("index", { dir });
      indexing = true;
    } catch(e) {
      console.error("Error indexing: ", e);
      indexing = false;
    }
  }

  const doSearch = async () => {
    let s = search.trim();
    if(s.length <= 3) {
      return;
    }

    searching = true;
    logs = [];
    searches = [{
      qry: s,
      files: [],
      evidence: [],
      answer: "",
      cfg: searchCfg
    }, ...searches];
    try {
      invoke("search", { qry: s, cfg: searchCfg })
    } catch(e) {
      console.error(`Error requesting search with qry: ${s}, error: `, e);
      setSearchFalse();
    }
  }

  onMount(() => {
    let window = getCurrentWindow();
    window.listen("status", ({ payload }) => {
      let s: StatusData = payload as StatusData; 
      let l = [...logs];
      l.push(s);
      logs = l;
    });

    window.listen("result", ({ payload}) => {
      let s: SearchResult = payload as SearchResult;
      s.cfg = searchCfg;
      searches[0] = s;
      searches = [...searches];
      setSearchFalse();
    });

    window.listen("error", ({ event, payload}) => {
      console.log(event, payload);
      searches[0] = { qry: search, files: [], evidence: [], answer: "Not Found!", cfg: searchCfg };
      searches = [...searches];
      setSearchFalse();
    });

    window.listen("indexing", ({ payload }) => {
      idxstatus = payload as IndexStat;
      if(idxstatus.progress >= 100) {
        let tout = setTimeout(() => {
          clearTimeout(tout);
          idxstatus = undefined;
          indexing = false;
        }, 3000);
      }
    });
  })

const setSearchFalse = () => {
  searching = false;
  search = "";
}

</script>

<div class="w-full h-full min-h-[100vh] flex flex-col">
  <div class="w-full p-8 flex flex-row">
    <div class="mb-2 w-8/12 flex flex-col">
      <div class="w-full">
        <input
          type="text" 
          class="block w-full p-4 text-gray-900 border border-gray-300 rounded-lg bg-gray-50 text-base focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
          placeholder="Type something and press Enter" bind:value={search} on:keyup={(kc) => { if(kc.key == "Enter") { doSearch() } }}
          >
      </div>
      <div class="flex flex-row gap-2 items-center py-1">
        <div class="flex flex-row gap-1 items-center">
          <span class="text-xs">BM25</span>
          <input type="checkbox" bind:checked={searchCfg.with_bm25}/>
        </div>
        <div class="flex flex-row gap-1 items-center">
          <span class="text-xs">Allow without Evidence</span>
          <input type="checkbox" bind:checked={searchCfg.allow_without_evidence}/>
        </div>
        <div class="flex flex-row gap-1 items-center">
          <span class="text-xs">Subqueries</span>
          <select bind:value={searchCfg.n_sub_qry} class="border text-sm rounded-lg block w-full p-2.5 bg-gray-700 border-gray-600 placeholder-gray-400 text-white focus:ring-blue-500 focus:border-blue-500">
            <option value={0}>0</option>
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={3}>3</option>
            <option value={4}>4</option>
          </select>
        </div>
        <div class="flex flex-row gap-1 items-center">
          <span class="text-xs">Adjacent</span>
          <select bind:value={searchCfg.k_adjacent} class="border text-sm rounded-lg block w-full p-2.5 bg-gray-700 border-gray-600 placeholder-gray-400 text-white focus:ring-blue-500 focus:border-blue-500">
            <option value={0}>0</option>
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={3}>3</option>
          </select>
        </div>
        <div class="flex flex-row gap-1 items-center">
          <span class="text-xs">Max Results</span>
          <select bind:value={searchCfg.max_result} class="border text-sm rounded-lg block w-full p-2.5 bg-gray-700 border-gray-600 placeholder-gray-400 text-white focus:ring-blue-500 focus:border-blue-500">
            <option value={2}>2</option>
            <option value={4}>4</option>
            <option value={6}>6</option>
            <option value={8}>8</option>
            <option value={10}>10</option>
          </select>
        </div>
      </div>
    </div>
    <button class="ml-auto button blue" on:click={folderPicker}>+ 📂</button>
  </div>
  <div class="grid grid-cols-[60%_40%] px-8">
    <div class="flex flex-col gap-4">
      {#each searches as search, i}
        <Search {search} searching={i == 0 ? searching : false}/>
      {/each}
    </div>
    {#if logs.length}
      <div class="flex flex-col px-4">
        <div class="text-md mb-2">Logs</div>
        {#each logs as log, i}
          <div class="flex flex-col mb-4 gap-2">
            <div class="flex flex-row items-center">
              <div class="text-sm font-medium">{i + 1}. {log.head}</div>
              {#if log.time_s || log.time_s == 0}
              <div class="text-xs font-bold ml-auto text-green-400 px-2">{ Math.round(log.time_s * 100) / 100 }s</div>
              {/if}
            </div>
            {#if log.body.length}
              <div class="text-xs">
                {@html log.body.replaceAll("\n", "<br>")}
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>
  {#if idxstatus && indexing}
  <div class="w-full h-full fixed bg-gray-900 bg-opacity-70 backdrop:blur-sm flex flex-col items-center justify-center top-0 left-0">
    <div class="w-1/2 flex flex-col bg-slate-700 p-8 rounded-xl shadow-xl">
      <div class="text-md">Indexing..</div>
      <div class="flex flex-row gap-2 items-center">
          <span class="font-medium text-xs">Files: </span>
          <span class="text-sm">{idxstatus.files}</span>
          <span class="font-medium text-xs">Pages: </span>
          <span class="text-sm">{idxstatus.pages}</span>
      </div>
      <div class="w-full h-6 bg-gray-800 rounded-md p-0.5 relative my-1">
        <div class="h-full rounded-md bg-green-700 float-right text-xs" style="width: {idxstatus.progress}%;">{Math.round(idxstatus.progress * 100) / 100}%</div>
      </div>
      <div class="text-xs">{idxstatus.msg}</div>
    </div>
  </div>
  {/if}
</div>