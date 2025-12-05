// Fleet Manager logic

const ws = new WebSocketClient();

// Initialize map
let fleetMap = null;
let vehicleMarkers = {};

function initializeMap() {
    // Center on Seoul, South Korea
    fleetMap = L.map('fleetMap').setView([37.5665, 126.9780], 11);

    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(fleetMap);
}

function updateMapMarkers(vehicles) {
    // Remove old markers
    Object.values(vehicleMarkers).forEach(marker => marker.remove());
    vehicleMarkers = {};

    // Add new markers for each vehicle
    vehicles.forEach(vehicle => {
        // Choose icon color based on status
        let iconColor = '#2563eb'; // blue for active
        if (vehicle.status === 'charging') iconColor = '#f59e0b'; // orange
        if (vehicle.status === 'idle') iconColor = '#64748b'; // gray
        if (vehicle.battery_percent < 20) iconColor = '#ef4444'; // red for low battery

        // Create custom icon
        const icon = L.divIcon({
            className: 'custom-marker',
            html: `<div style="background: ${iconColor}; width: 30px; height: 30px; border-radius: 50%;
                   border: 3px solid white; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                   display: flex; align-items: center; justify-content: center;
                   color: white; font-weight: bold; font-size: 12px;">
                   ${vehicle.is_real ? '⚡' : vehicle.id.slice(-1)}
                   </div>`,
            iconSize: [30, 30],
            iconAnchor: [15, 15]
        });

        // Create marker
        const marker = L.marker([vehicle.latitude, vehicle.longitude], { icon })
            .addTo(fleetMap);

        // Add popup with vehicle info
        const popupContent = `
            <div style="font-family: sans-serif;">
                <h3 style="margin: 0 0 8px 0; color: #2563eb;">${vehicle.id} ${vehicle.is_real ? '⚡' : ''}</h3>
                <p style="margin: 4px 0;"><strong>Status:</strong> ${vehicle.status.toUpperCase()}</p>
                <p style="margin: 4px 0;"><strong>Battery:</strong> ${vehicle.battery_percent}%</p>
                <p style="margin: 4px 0;"><strong>Product:</strong> ${vehicle.product_tag}</p>
                <p style="margin: 4px 0;"><strong>Route:</strong><br>${vehicle.departure} → ${vehicle.destination}</p>
            </div>
        `;
        marker.bindPopup(popupContent);

        // Store marker reference
        vehicleMarkers[vehicle.id] = marker;
    });

    // Auto-fit map to show all markers
    if (vehicles.length > 0) {
        const bounds = L.latLngBounds(vehicles.map(v => [v.latitude, v.longitude]));
        fleetMap.fitBounds(bounds, { padding: [50, 50] });
    }
}

// Initialize map when page loads
window.addEventListener('load', () => {
    initializeMap();
});

// Update fleet display
ws.on('fleet', (data) => {
    updateFleetSummary(data.summary);
    updateVehicleGrid(data.vehicles);
    updateMapMarkers(data.vehicles);
});

function updateFleetSummary(summary) {
    document.getElementById('totalVehicles').textContent = summary.total_vehicles;
    document.getElementById('activeCount').textContent = summary.active_count;
    document.getElementById('idleCount').textContent = summary.idle_count;
    document.getElementById('chargingCount').textContent = summary.charging_count;
    document.getElementById('lowBatteryCount').textContent = summary.low_battery_count;
}

function updateVehicleGrid(vehicles) {
    const grid = document.getElementById('vehicleGrid');
    grid.innerHTML = vehicles.map(v => createVehicleCard(v)).join('');
}

function createVehicleCard(vehicle) {
    const batteryClass = vehicle.battery_percent < 20 ? 'low' : vehicle.battery_percent < 50 ? 'medium' : '';
    const realBadge = vehicle.is_real ? '<span class="real-badge">⚡ REAL</span>' : '';

    return `
        <div class="vehicle-card">
            <div class="vehicle-header">
                <div class="vehicle-id">${vehicle.id}</div>
                ${realBadge}
            </div>
            <div class="vehicle-info">
                <div class="info-row">
                    <span class="info-label">Model</span>
                    <span class="info-value">${vehicle.model}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Product</span>
                    <span class="info-value">${vehicle.product_tag}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Journey</span>
                    <span class="info-value">${vehicle.departure} → ${vehicle.destination}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Location</span>
                    <span class="info-value">${vehicle.latitude.toFixed(4)}°N, ${vehicle.longitude.toFixed(4)}°E</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Battery</span>
                    <span class="info-value">${vehicle.battery_percent}%</span>
                </div>
                <div class="battery-bar">
                    <div class="battery-fill ${batteryClass}" style="width: ${vehicle.battery_percent}%"></div>
                </div>
                <div class="info-row">
                    <span class="info-label">Status</span>
                    <span class="status-badge ${vehicle.status}">${vehicle.status.toUpperCase()}</span>
                </div>
            </div>
            <div class="vehicle-actions">
                <button class="btn-secondary" onclick="editVehicle('${vehicle.id}')" ${vehicle.is_real ? 'disabled' : ''}>Edit</button>
                <button class="btn-secondary btn-danger" onclick="deleteVehicle('${vehicle.id}')" ${vehicle.is_real ? 'disabled' : ''}>Delete</button>
            </div>
        </div>
    `;
}

async function showAddVehicleForm() {
    const id = prompt('Vehicle ID (e.g., TB3-004):');
    if (!id) return;

    const product_tag = prompt('Product Tag:', 'Electronics');
    const departure = prompt('Departure:', 'Incheon Airport Warehouse');
    const destination = prompt('Destination:', 'Gangnam Distribution Hub');

    try {
        await apiRequest('/api/vehicles', {
            method: 'POST',
            body: JSON.stringify({ id, product_tag, departure, destination })
        });
        alert('Vehicle added successfully!');
    } catch (error) {
        alert('Error adding vehicle: ' + error.message);
    }
}

async function editVehicle(id) {
    const product_tag = prompt('New Product Tag:');
    if (!product_tag) return;

    try {
        await apiRequest(`/api/vehicles/${id}`, {
            method: 'PUT',
            body: JSON.stringify({ product_tag })
        });
        alert('Vehicle updated!');
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function deleteVehicle(id) {
    if (!confirm(`Delete vehicle ${id}?`)) return;

    try {
        await apiRequest(`/api/vehicles/${id}`, { method: 'DELETE' });
        alert('Vehicle deleted!');
    } catch (error) {
        alert('Error: ' + error.message);
    }
}
