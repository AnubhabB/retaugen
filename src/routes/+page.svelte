<script lang="ts">
  import { open } from '@tauri-apps/plugin-dialog';
  import { invoke } from "@tauri-apps/api/core";

  let search: string;

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
    let s = search.trim();
    if(s.length <= 3) {
      return;
    }

    try {
      invoke("search", { qry: s })
    } catch(e) {
      console.error(`Error requesting search with qry: ${s}, error: `, e);
    }
  }

</script>

<div class="w-full h-full min-h-[100vh] flex flex-col">
  <div class="w-full p-8 flex flex-row">
    <div class="mb-2 w-8/12">
      <input
      type="text" 
      class="block w-full p-4 text-gray-900 border border-gray-300 rounded-lg bg-gray-50 text-base focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
      placeholder="Type something and press Enter" bind:value={search} on:keyup={(kc) => { if(kc.key == "Enter") { doSearch() } }}
      >
    </div>
    <button class="ml-auto button blue" on:click={folderPicker}>+ 📂</button>
  </div>
</div>