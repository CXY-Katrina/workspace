import fs from 'node:fs/promises';
import path from 'node:path';
import {PresentationFile,FileBlob} from 'file:///C:/Users/Katrina/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
process.env.RUNTIME_NODE_MODULES='C:/Users/Katrina/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const SKILL='C:/Users/Katrina/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const {finalizePresentation}=await import('file:///'+SKILL+'/container_tools/artifact_tool_utils.mjs');
const root=path.resolve('inference_metrics'),candidate=path.join(root,'.pptx-build/candidate.pptx');
// Import directly from the bundled entry point; this installation has no package.json.
const check=await PresentationFile.importPptx(await FileBlob.load(candidate));
if(check.slides.items.length!==2)throw new Error('Expected two slides');
await fs.mkdir(path.join(root,'output'),{recursive:true});
const finalPath=path.join(root,'output/推理性能指标-两页可编辑.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,pythonExecutable:'C:/Users/Katrina/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:2,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'design',families:['Microsoft YaHei']},verifyArtifactToolImport:false,receiptPath:path.join(root,'.pptx-build/validation.json')});
console.log(JSON.stringify(result));
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<2;i++){
 const blob=await finalDeck.export({slide:finalDeck.slides.items[i],format:'png',scale:1.5});
 await fs.writeFile(path.join(root,`.pptx-build/final-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
console.log(finalPath);
