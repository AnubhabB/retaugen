<script lang="ts">
  import { open } from '@tauri-apps/plugin-dialog';
  import { invoke } from "@tauri-apps/api/core";

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

</script>

<div class="w-full h-full min-h-[100vh] flex flex-col">
  <div class="w-full p-8 flex flex-row">
    <div>Search input goes here!</div>
    <button class="ml-auto button blue" on:click={folderPicker}>+ 📂</button>
  </div>
</div>