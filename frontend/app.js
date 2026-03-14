const videoElement = document.getElementById('camera-stream');
const canvasElement = document.getElementById('capture-canvas');
const startBtn = document.getElementById('start-btn');
const captureBtn = document.getElementById('capture-btn');
const statusInd = document.getElementById('status-indicator');
const API_TIMEOUT_MS = 20000;

const emptyState = document.getElementById('empty-state');
const labelPreview = document.getElementById('label-preview');

let stream = null;
let isProcessing = false;
let jsonPayload = null;

startBtn.addEventListener('click', async () => {
    if (stream) {
        stopCamera();
    } else {
        await startCamera();
    }
});

captureBtn.addEventListener('click', takePhoto);

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } }
        });
        videoElement.srcObject = stream;

        startBtn.textContent = 'Stop Camera';
        startBtn.classList.replace('primary', 'error') || startBtn.classList.add('error');
        captureBtn.disabled = false;
        statusInd.textContent = 'Camera active. Ready to capture.';
    } catch (err) {
        console.error('Error accessing camera:', err);
        statusInd.textContent = 'Error: Could not access camera. Please allow permissions.';
    }
}

function stopCamera() {
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        videoElement.srcObject = null;
        stream = null;
    }
    isProcessing = false;

    startBtn.textContent = 'Start Camera';
    startBtn.classList.replace('error', 'primary');
    captureBtn.disabled = true;

    statusInd.textContent = 'Camera stopped.';
}

async function takePhoto() {
    if (!stream || isProcessing) return;

    isProcessing = true;
    captureBtn.disabled = true;
    captureBtn.textContent = 'Scanning...';
    statusInd.textContent = 'Capturing burst...';
    console.log('Burst capture started');

    try {
        const frames = await captureBurstFrames(3, 70); // 3 frames, 70ms apart (~<0.3s)
        if (!frames.length) throw new Error('No frames captured');
        frames.forEach((f, idx) => console.log(`Frame ${idx + 1} sharpness: ${f.sharpness.toFixed(2)}`));
        frames.sort((a, b) => b.sharpness - a.sharpness);
        const best = frames[0];
        console.log('Selected frame sharpness:', best.sharpness.toFixed(2));
        statusInd.textContent = 'Sending to scanner...';
        await postFrameToGateway(best.blob);
    } catch (e) {
        console.error(e);
        statusInd.textContent = 'Capture failed.';
    } finally {
        isProcessing = false;
        captureBtn.disabled = false;
        captureBtn.textContent = 'Take Photo';
    }
}

function calculateSharpness(imageData) {
    const data = imageData.data;
    const width = imageData.width;
    const height = imageData.height;
    const gray = new Uint8Array(width * height);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
        gray[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
    let sum = 0;
    let count = 0;
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const i = y * width + x;
            const top = (y - 1) * width + x;
            const bottom = (y + 1) * width + x;
            const left = y * width + (x - 1);
            const right = y * width + (x + 1);
            const value = gray[top] + gray[bottom] + gray[left] + gray[right] - 4 * gray[i];
            sum += value * value; // squared Laplacian
            count++;
        }
    }
    return sum / Math.max(1, count);
}

async function captureBurstFrames(count, intervalMs) {
    const frames = [];
    const rawWidth = videoElement.videoWidth;
    const rawHeight = videoElement.videoHeight;
    if (!rawWidth || !rawHeight) return frames;

    canvasElement.width = rawWidth;
    canvasElement.height = rawHeight;
    const ctx = canvasElement.getContext('2d');

    for (let i = 0; i < count; i++) {
        ctx.drawImage(videoElement, 0, 0, rawWidth, rawHeight);
        const imageData = ctx.getImageData(0, 0, rawWidth, rawHeight);
        const sharpness = calculateSharpness(imageData);
        const blob = await new Promise(resolve => canvasElement.toBlob(resolve, 'image/jpeg', 0.95));
        if (blob) {
            frames.push({ blob, sharpness });
        }
        if (i < count - 1) {
            await new Promise(r => setTimeout(r, intervalMs));
        }
    }
    return frames;
}

async function postFrameToGateway(blob) {
    const formData = new FormData();
    formData.append('file', blob, 'capture.jpg');

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT_MS);

    try {
        console.log('Scan request sent');
        const response = await fetch("/scan", {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });

        if (!response.ok) throw new Error("HTTP error " + response.status);

        const data = await response.json();
        console.log('Scan response received', data);

        if (data.structured_data) {
            statusInd.textContent = data.status === "success"
                ? 'Label data ready.'
                : 'Partial label data returned.';
            renderPreview(data);
        } else {
            statusInd.textContent = 'No label data returned.';
        }
    } catch (e) {
        console.warn("API check failed", e);
        statusInd.textContent = e.name === 'AbortError'
            ? `API timeout (${API_TIMEOUT_MS / 1000}s).`
            : 'Connection error reaching API.';
    } finally {
        clearTimeout(timeoutId);
        isProcessing = false;
        captureBtn.disabled = false;
        captureBtn.textContent = 'Take Photo';
    }
}

function renderPreview(data) {
    emptyState.classList.add('hidden');
    labelPreview.classList.remove('hidden');

    const parsed = data.parsed || {};
    const structured = data.structured_data || {};

    document.getElementById('lbl-part').textContent = parsed.part || structured.part_number || '-';
    document.getElementById('lbl-qty').textContent = parsed.qty || structured.quantity || '-';
    document.getElementById('lbl-lot').textContent = parsed.ven_lot_no || structured.vendor_lot || '-';
    document.getElementById('lbl-vendor').textContent = parsed.vendor || structured.vendor || 'UNKNOWN';
    document.getElementById('lbl-desc').textContent = structured.description || '-';
    document.getElementById('lbl-hu').textContent = parsed.barcode || structured.hu || '-';
    document.getElementById('lbl-ibd').textContent = parsed.ibd_no || structured.ibd || '-';

    document.getElementById('lbl-invoice').textContent = parsed.supplier_invoice || structured.supplier_invoice || '-';
    document.getElementById('lbl-msd').textContent = parsed.msd_level || structured.msd_level || '-';
    document.getElementById('lbl-msd-date').textContent = parsed.msd_date || structured.msd_date || '-';

    document.getElementById('meta-engine').textContent = data.meta.engine_used;
    document.getElementById('meta-time').textContent = data.meta.processing_time;

    try {
        bwipjs.toCanvas('datamatrix-canvas', {
            bcid: 'datamatrix',
            text: structured.datamatrix || `PRN${parsed.part || structured.part_number || '-'}LOT${parsed.ven_lot_no || structured.vendor_lot || '-'}QTY${parsed.qty || structured.quantity || '-'}`,
            scale: 3,
            height: 10,
            width: 10,
            includetext: false,
            textxalign: 'center',
        });
    } catch (e) {
        console.error("Barcode generation error:", e);
    }

    jsonPayload = {
        raw_text: data.raw_text || structured.raw_text || '',
        blocks: data.blocks || [],
        parsed,
        structured_data: structured,
    };
}

document.getElementById('download-json').addEventListener('click', () => {
    if (!jsonPayload) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(jsonPayload, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "scanned_data.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
});

document.getElementById('download-zpl').addEventListener('click', async () => {
    if (!jsonPayload || !jsonPayload.structured_data || !jsonPayload.structured_data.zpl) return;
    try {
        await navigator.clipboard.writeText(jsonPayload.structured_data.zpl);
        statusInd.textContent = 'ZPL copied to clipboard';
    } catch (e) {
        const blob = new Blob([jsonPayload.structured_data.zpl], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'label.zpl';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        statusInd.textContent = 'ZPL downloaded';
    }
});
