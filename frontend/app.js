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
    const labelFields = data.label_fields || structured.label_fields || {};

    const partValue = labelFields.part || parsed.part || structured.part_number || '-';
    const qtyValue = labelFields.qty || parsed.qty || structured.quantity || '-';
    const packQty = labelFields.pack_qty;
    const qtyDisplay = labelFields.qty_display || ((packQty && qtyValue !== "-") ? `${qtyValue} /${packQty} EA` : qtyValue);
    const lotValue = labelFields.vendor_lot || parsed.ven_lot_no || structured.vendor_lot || '-';
    const ibdValue = labelFields.ibd_no || parsed.ibd_no || structured.ibd || '-';
    const huValue = labelFields.barcode_number || parsed.barcode || structured.hu || '-';
    const invoiceValue = labelFields.supplier_invoice || parsed.supplier_invoice || structured.supplier_invoice || '-';
    const msdLevelValue = labelFields.msd_level || parsed.msd_level || structured.msd_level || '-';
    const msdDateValue = labelFields.msd_date || parsed.msd_date || structured.msd_date || '-';

    const vendorText = (labelFields.vendor_code && labelFields.vendor_display)
        ? `${labelFields.vendor_code} / ${labelFields.vendor_display}`
        : (parsed.vendor || structured.vendor || 'UNKNOWN');

    const descriptionLines = Array.isArray(labelFields.description_lines) && labelFields.description_lines.length
        ? labelFields.description_lines
        : [structured.description || '-'];

    document.getElementById('lbl-part').textContent = partValue;
    document.getElementById('lbl-qty').textContent = qtyDisplay;
    document.getElementById('lbl-lot').textContent = lotValue;
    document.getElementById('lbl-vendor').textContent = vendorText;
    document.getElementById('lbl-desc-1').textContent = descriptionLines[0] || '-';
    document.getElementById('lbl-desc-2').textContent = descriptionLines[1] || '';
    document.getElementById('lbl-desc-3').textContent = descriptionLines[2] || '';
    document.getElementById('lbl-hu').textContent = huValue;
    document.getElementById('lbl-ibd').textContent = ibdValue;

    document.getElementById('lbl-invoice').textContent = invoiceValue;
    document.getElementById('lbl-msd').textContent = msdLevelValue;
    document.getElementById('lbl-msd-date').textContent = msdDateValue;

    document.getElementById('meta-engine').textContent = data.meta.engine_used;
    document.getElementById('meta-time').textContent = data.meta.processing_time;

    try {
        const qrPayload = labelFields.qr_payload
            || structured.datamatrix
            || `PRN${partValue}LOT${lotValue}QTY${qtyValue}`;
        bwipjs.toCanvas('qr-canvas', {
            bcid: 'qrcode',
            text: qrPayload,
            scale: 2,
            eclevel: 'L',
            includetext: false,
        });
    } catch (e) {
        console.error("Barcode generation error:", e);
    }

    jsonPayload = {
        raw_text: data.raw_text || structured.raw_text || '',
        blocks: data.blocks || [],
        parsed,
        label_fields: labelFields,
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


