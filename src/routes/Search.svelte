<script lang="ts">
    import Thumb from "./Thumb.svelte";
    import type { SearchResult } from "./types";
    import { open, FileHandle, BaseDirectory, readFile, readTextFile } from '@tauri-apps/plugin-fs';

    export let search: SearchResult,
        searching: boolean = false;

    enum FileKind {
        Pdf,
        Txt,
        Html
    }

    console.log(search.files);
    let fileData: string|undefined,
        kind: FileKind;

    const openFile = async (path: string, page: number|undefined) => {
        console.log("Read file: ", path, page);

        if(path.endsWith(".pdf")) {
            // console.log(buf.length);
            const fread = await readFile(path);
            const blob = new Blob([fread], { type: 'application/pdf' });
            // Create a URL for the Blob
            fileData  = `${URL.createObjectURL(blob)}#page=${page||1}`;
            kind = FileKind.Pdf;
        } else {
            fileData = await readTextFile(path);
            kind = path.endsWith(".txt") ? FileKind.Txt : FileKind.Html;
        }
    }
</script>

<div class="grid grid-cols-[24px_calc(100%-24px)]">
    <div>Q.</div>
    <span class="flex flex-row">
        <div class="font-medium text-gray-400">{search.qry}</div>
        {#if search.elapsed}
        <div class="font-medium text-green-400 text-xs ml-auto">[{Math.round(search.elapsed * 100) / 100}s]</div>
        {/if}
    </span>
    <div>A.</div>
    <div class="flex flex-col">
        {#if searching}
            <div>Searching ..</div>
        {/if}
        <div class="text-md">{ searching ?  "" : search.answer || "Not Found"}</div>
        {#if search.answer.length && (!search.evidence.length || !search.files.length)}
            <div class="text-red-400 text-sm font-bold">This answer was generated without evidence!</div>
        {/if}
        {#if search.evidence.length}
            <div class="flex flex-col py-2">
                <div class="text-sm font-bold text-gray-400">Evidence</div>
                <div class="flex flex-col gap-2">
                    {#each search.evidence as e}
                        <span class="text-xs">
                            • {e.text} <span role="link" tabindex="-1" class="text-xs font-bold text-blue-400 cursor-pointer" on:keyup|preventDefault on:click={() => { openFile(e.file, e.page) }}>Source</span>
                        </span>
                    {/each}
                </div>
            </div>
        {/if}
    </div>
</div>

{#if fileData}
<div class="w-full h-full fixed bg-gray-900 bg-opacity-70 backdrop:blur-sm flex flex-col items-center justify-center top-0 left-0">
    <div class="w-11/12 h-[100vh] bg-white relative">
        <button class="fixed right-2 top-2" on:click={async () => { fileData = undefined; }}>Close</button>
        {#if kind == FileKind.Txt}
        <div class="p-4">
            {@html fileData?.replaceAll("\n", "<br>")}
        </div>
        {:else if kind == FileKind.Html}
            {@html fileData}
        {:else}
            <embed src="{fileData}" style="width: 100%; height: 100vh"/>
        {/if}
    </div>
</div>
{/if}