// Driver interface logic

const ws = new WebSocketClient();
let currentMode = 'autonomous';
let currentSpeed = 0.15;

// Mode switching
document.querySelectorAll('input[name="mode"]').forEach(radio => {
    radio.addEventListener('change', async (e) => {
        currentMode = e.target.value;
        await apiRequest('/api/mode', {
            method: 'POST',
            body: JSON.stringify({ mode: currentMode })
        });
        updateModeUI();
    });
});

function updateModeUI() {
    const manualControls = document.getElementById('manualControls');
    const currentModeEl = document.getElementById('currentMode');

    if (currentMode === 'manual') {
        manualControls.style.display = 'block';
        currentModeEl.textContent = 'Manual';
    } else {
        manualControls.style.display = 'none';
        currentModeEl.textContent = 'Autonomous';
    }
}

// Manual controls
const controls = {
    forward: { linear: currentSpeed, angular: 0 },
    backward: { linear: -currentSpeed, angular: 0 },
    left: { linear: 0, angular: 0.5 },
    right: { linear: 0, angular: -0.5 },
    stop: { linear: 0, angular: 0 }
};

function sendControl(action) {
    if (currentMode !== 'manual') return;

    const control = { ...controls[action] };
    if (action === 'forward' || action === 'backward') {
        control.linear = action === 'forward' ? currentSpeed : -currentSpeed;
    }
    ws.send('control', control);
}

// Button controls
document.querySelectorAll('.control-btn').forEach(btn => {
    btn.addEventListener('mousedown', () => sendControl(btn.dataset.action));
    btn.addEventListener('mouseup', () => sendControl('stop'));
    btn.addEventListener('touchstart', (e) => { e.preventDefault(); sendControl(btn.dataset.action); });
    btn.addEventListener('touchend', (e) => { e.preventDefault(); sendControl('stop'); });
});

// Keyboard controls
document.addEventListener('keydown', (e) => {
    if (currentMode !== 'manual') return;

    const key = e.key.toLowerCase();
    if (key === 'w' || key === 'arrowup') sendControl('forward');
    else if (key === 's' || key === 'arrowdown') sendControl('backward');
    else if (key === 'a' || key === 'arrowleft') sendControl('left');
    else if (key === 'd' || key === 'arrowright') sendControl('right');
});

document.addEventListener('keyup', (e) => {
    if (currentMode !== 'manual') return;
    const key = e.key.toLowerCase();
    if (['w', 's', 'a', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(key)) {
        sendControl('stop');
    }
});

// Speed control
const speedSlider = document.getElementById('speedSlider');
const speedValue = document.getElementById('speedValue');

speedSlider.addEventListener('input', (e) => {
    currentSpeed = parseFloat(e.target.value);
    speedValue.textContent = currentSpeed.toFixed(2);
});

// Emergency stop
document.getElementById('emergencyStop').addEventListener('click', () => {
    ws.send('control', { linear: 0, angular: 0 });
});

// Camera feed
ws.on('camera', (data) => {
    document.getElementById('cameraFeed').src = data.frame;
});

// Status updates
ws.on('status', (data) => {
    document.getElementById('statusState').textContent = (data.state || '-').toUpperCase();
    document.getElementById('statusLinear').textContent = data.linear ? `${data.linear.toFixed(3)} m/s` : '- m/s';
    document.getElementById('statusAngular').textContent = data.angular ? `${data.angular.toFixed(3)} rad/s` : '- rad/s';
    document.getElementById('statusPedestrian').textContent = data.pedestrian_status || data.pedestrian || '-';
    document.getElementById('statusLane').textContent = data.lane_status || data.lane || '-';
    document.getElementById('statusBoundary').textContent = data.boundary_status || '-';
    document.getElementById('statusFPS').textContent = data.fps || '-';
});

updateModeUI();
