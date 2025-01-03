// import { FilesetResolver, ObjectDetector } from "@mediapipe/tasks-vision";
// import { FilesetResolver, ObjectDetector } from "/node_modules/.pnpm/@mediapipe+tasks-vision@0.10.12/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";

import { FilesetResolver, ObjectDetector } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/+esm'


let objectDetector;
let detections = [];        

// Load the model and set options
async function initializeObjectDetector() {

    // Access the camera, global navigator object
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })

    document.getElementById('camera-stream').srcObject = stream;

    // const vision = await FilesetResolver.forVisionTasks(
    //     "/node_modules/@mediapipe/tasks-vision/wasm"
    // );

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
    );

    objectDetector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "public/i320_8x100.tflite",
        },
        scoreThreshold: 0.4,
        runningMode: "VIDEO",

    });

    renderLoop();
}

let lastVideoTime = -1;
function renderLoop() {
    const video = document.getElementById("camera-stream");

    if (video.currentTime !== lastVideoTime) {
        const output = objectDetector.detectForVideo(video, Date.now());
        processResults(output);
        lastVideoTime = video.currentTime;
    }

    requestAnimationFrame(() => renderLoop() );
}

function processResults(output) {
    const container = document.getElementById("container");

    for (let d of detections) {
        container.removeChild(d);
    }

    detections = [];

    for (let o of output.detections) {
        const x = o.boundingBox.originX;
        const y = o.boundingBox.originY;
        const w = o.boundingBox.width;
        const h = o.boundingBox.height;
        const sc = o.categories[0].score;
        const lb = o.categories[0].categoryName;

        const box = boundingBox(x, y, w, h, lb, sc);
        detections.push(box);
    }

    console.log(detections);
    container.append(...detections);
}


function boundingBox(x, y, w, h, lb, sc) {

    if (!boundingBox.static) boundingBox.static = 0;

    const box = document.createElement("div");
    box.id = `object-${boundingBox.static++}`;
    box.style = `
        position: absolute;
        top: ${y-20}px;
        left: ${x}px;
        width: ${w}px;
        height: ${h+20}px;
        border: 4px solid #7158e2;
    `;

    const info = document.createElement("div");
    info.style = `
        display: flex;
        justify-content: space-between;
        border-bottom: 4px solid #7158e2;
        background-color: #7158e2;
        padding: 1px 3px;
        color: #d2dae2;
        font-family: arial;
        font-size: 14px;
    `;

    const label = document.createElement('span');
    label.append(lb);

    const score = document.createElement('span');
    score.append(`${Math.ceil(sc * 100)}%`);

    info.append(...[label, score]);
    box.append(info);

    return box;
}

initializeObjectDetector();
