<script lang="ts">
  import { open } from '@tauri-apps/plugin-dialog';
  import { invoke } from "@tauri-apps/api/core";
  import { getCurrentWindow } from '@tauri-apps/api/window';
    import { onMount } from 'svelte';

  interface SearchConfig {
    with_bm25: boolean,
    max_result: number,
    ann_cutoff: number,
    n_sub_qry: number,
    k_adjacent: number,
    relevance_cutoff: number,
  }

  let searchCfg: SearchConfig = {
    with_bm25: true,
    max_result: 8,
    ann_cutoff: 0.75,
    n_sub_qry: 3,
    k_adjacent: 1,
    relevance_cutoff: 5.,
  };

  let search: string;

  let logs: string[] = [];

  const folderPicker = async () => {
    const dir = await open({
      multiple: false,
      directory: true,
    });
    
    if(!dir)
      return;

    try {
      invoke("index", { dir })
    } catch(e) {
      console.error("Error indexing: ", e);
    }
  }

  const doSearch = async () => {
    logs = [];
    let s = search.trim();
    if(s.length <= 3) {
      return;
    }

    try {
      invoke("search", { qry: s, cfg: searchCfg })
    } catch(e) {
      console.error(`Error requesting search with qry: ${s}, error: `, e);
    }
  }

  onMount(() => {
    console.log("Window ready!");
    let window = getCurrentWindow();
    window.listen("status", ({ payload}) => {
      let s = (payload as string).replaceAll("\n", "<br>");
      console.log("Comes here ..");
      let l = [...logs];
      l.push(s);
      logs = l;
    })

    window.listen("result", ({ event, payload}) => {
      console.log(event, payload)
    })

    window.listen("error", ({ event, payload}) => {
      console.log(event, payload)
    })
  })

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
      <div class="flex flex-row gap-2">

      </div>
    </div>
    <button class="ml-auto button blue" on:click={folderPicker}>+ 📂</button>
  </div>
  <div class="grid grid-cols-[60%_40%]">
    <div></div>
    <div class="flex flex-col">
      {#each logs as log}
        <div>{@html log}</div>
      {/each}
    </div>
  </div>
</div>