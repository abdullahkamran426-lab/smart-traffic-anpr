-- vehicle_log: Core table — one row per vehicle crossing event
CREATE TABLE IF NOT EXISTS vehicle_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_type    TEXT    NOT NULL CHECK(vehicle_type IN ('car','motorcycle','bus','truck')),
    plate_number    TEXT,                    -- NULL if OCR failed or confidence too low
    ocr_confidence  REAL CHECK(ocr_confidence BETWEEN 0 AND 1),
    video_source    TEXT    DEFAULT '',
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- session_log: Track each pipeline run
CREATE TABLE IF NOT EXISTS session_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_start   DATETIME NOT NULL,
    session_end     DATETIME,
    video_source    TEXT,
    total_vehicles  INTEGER DEFAULT 0,
    notes           TEXT
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_timestamp     ON vehicle_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_vehicle_type  ON vehicle_log(vehicle_type);
CREATE INDEX IF NOT EXISTS idx_plate         ON vehicle_log(plate_number);

-- Sample records
INSERT INTO vehicle_log (vehicle_type, plate_number, ocr_confidence, video_source)
VALUES
  ('truck',       'ABC123', 0.92, 'sample_traffic_1.mp4'),
  ('car',         'XYZ456', 0.87, 'sample_traffic_1.mp4'),
  ('motorcycle',   NULL,     NULL, 'sample_traffic_1.mp4'),
  ('bus',         'DEF789', 0.91, 'sample_traffic_1.mp4');
