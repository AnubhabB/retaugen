<script lang="ts">
    import type { SearchResult } from "./types";

    export let search: SearchResult,
        searching: boolean = false;

    console.log(search.files);
</script>

<div class="grid grid-cols-[24px_calc(100%-24px)]">
    <div>Q.</div>
    <div class="font-medium text-gray-400">{search.qry}</div>
    <div>A.</div>
    <div class="flex flex-col">
        {#if searching}
            <div>Searching ..</div>
        {/if}
        <div class="text-md">{search.answer || searching ? "" : "Not Found"}</div>
        {#if search.answer.length && !search.evidence.length}
            <div class="text-red-400 text-sm font-bold">This answer was generated without evidence!</div>
        {/if}
        {#if search.evidence.length}
            <div class="flex flex-col py-2">
                <div class="text-sm font-bold text-gray-400">Evidence</div>
                <div class="flex flex-col">
                    {#each search.evidence as e}
                        <span class="text-xs">• {e}</span>
                    {/each}
                </div>
            </div>
        {/if}
        {#if search.files.length}
        <div class="flex flex-col py-2">
            <div class="text-sm font-bold text-gray-400">Files</div>
            <div class="flex flex-row">
                {#each search.files as f }
                    <div class="aspect-square overflow-hidden">sf</div>
                {/each}
            </div>
        </div>
        {/if}
    </div>
</div>