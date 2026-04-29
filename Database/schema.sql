-- ═══════════════════════════════════════════════════════════════
-- BASE DE DATOS: MCFC Digital Twin
-- Motor: PostgreSQL 14+
-- Descripción: Almacenamiento de datos experimentales de celdas
--              de combustible de carbonatos fundidos (MCFC) y
--              parámetros del modelo semi-empírico Nernst+pérdidas.
-- ═══════════════════════════════════════════════════════════════

-- Limpieza (solo desarrollo)
DROP TABLE IF EXISTS mediciones CASCADE;
DROP TABLE IF EXISTS parametros_modelo CASCADE;
DROP TABLE IF EXISTS experimentos CASCADE;
DROP TYPE IF EXISTS fuente_datos CASCADE;

-- ───────────────────────────────────────────────────────────────
-- TIPO ENUM: Origen de los datos experimentales
-- ───────────────────────────────────────────────────────────────
CREATE TYPE fuente_datos AS ENUM ('warsaw_ut', 'udec_lab');

-- ───────────────────────────────────────────────────────────────
-- TABLA 1: experimentos
-- Condiciones experimentales (entrada del experimento)
-- ───────────────────────────────────────────────────────────────
CREATE TABLE experimentos (
    id_experimento      SERIAL          PRIMARY KEY,
    fuente              fuente_datos    NOT NULL DEFAULT 'warsaw_ut',
    
    -- Temperatura de operación (°C)
    T                   SMALLINT        NOT NULL
        CHECK (T BETWEEN 500 AND 750),
    
    -- Composición ánodo (fracciones molares relativas)
    H2a                 DOUBLE PRECISION CHECK (H2a >= 0),
    H2Oa                DOUBLE PRECISION CHECK (H2Oa >= 0),
    N2a                 DOUBLE PRECISION CHECK (N2a >= 0),
    CO                  DOUBLE PRECISION CHECK (CO >= 0),
    CH4                 DOUBLE PRECISION CHECK (CH4 >= 0),
    CO2a                DOUBLE PRECISION CHECK (CO2a >= 0),
    
    -- Composición cátodo (fracciones molares relativas)
    O2c                 DOUBLE PRECISION CHECK (O2c >= 0),
    N2c                 DOUBLE PRECISION CHECK (N2c >= 0),
    CO2c                DOUBLE PRECISION CHECK (CO2c >= 0),
    H2Oc                DOUBLE PRECISION CHECK (H2Oc >= 0),
    
    -- Parámetros de materiales del electrodo y electrolito
    delta_Nia           DOUBLE PRECISION,
    rho_a               DOUBLE PRECISION,
    delta_LiKe          DOUBLE PRECISION,
    delta_NiOc          DOUBLE PRECISION,
    rho_c               DOUBLE PRECISION,
    
    -- Auditoría
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE experimentos IS 
    'Condiciones experimentales únicas de cada ensayo MCFC';
COMMENT ON COLUMN experimentos.T IS 'Temperatura de operación en °C';
COMMENT ON COLUMN experimentos.fuente IS 
    'Origen de los datos: warsaw_ut (Prof. Milewski) o udec_lab';

-- ───────────────────────────────────────────────────────────────
-- TABLA 2: parametros_modelo
-- Parámetros calculados del modelo semi-empírico Nernst+pérdidas
-- Relación 1:1 con experimentos
-- ───────────────────────────────────────────────────────────────
CREATE TABLE parametros_modelo (
    id_experimento      INTEGER         PRIMARY KEY,
    E_max               DOUBLE PRECISION,
    i_max               DOUBLE PRECISION,
    r_1                 DOUBLE PRECISION,   -- coeficiente óhmico
    r_2                 DOUBLE PRECISION,   -- coeficiente activación
    n_H2_a_in           DOUBLE PRECISION,   -- flujo molar H2 entrada
    
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_parametros_experimento
        FOREIGN KEY (id_experimento)
        REFERENCES experimentos(id_experimento)
        ON DELETE CASCADE
);

COMMENT ON TABLE parametros_modelo IS 
    'Parámetros del modelo Nernst+pérdidas por experimento';

-- ───────────────────────────────────────────────────────────────
-- TABLA 3: mediciones
-- Puntos de la curva de polarización (muchos por experimento)
-- ───────────────────────────────────────────────────────────────
CREATE TABLE mediciones (
    id_medicion         BIGSERIAL       PRIMARY KEY,
    id_experimento      INTEGER         NOT NULL,
    
    i_densidad          DOUBLE PRECISION NOT NULL
        CHECK (i_densidad >= 0),       -- A/cm²
    voltaje             DOUBLE PRECISION NOT NULL
        CHECK (voltaje >= 0),          -- V
    eta                 DOUBLE PRECISION
        CHECK (eta >= 0 AND eta <= 1), -- eficiencia [0,1]
    
    -- Timestamp de adquisición (relevante para datos en tiempo real del lab UdeC)
    timestamp_medicion  TIMESTAMPTZ,
    
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_mediciones_experimento
        FOREIGN KEY (id_experimento)
        REFERENCES experimentos(id_experimento)
        ON DELETE CASCADE
);

COMMENT ON TABLE mediciones IS 
    'Puntos individuales de las curvas de polarización (E vs. i)';
COMMENT ON COLUMN mediciones.timestamp_medicion IS 
    'Timestamp de adquisición — NULL para datos históricos, poblado para datos en tiempo real';

-- ───────────────────────────────────────────────────────────────
-- ÍNDICES ESTRATÉGICOS
-- ───────────────────────────────────────────────────────────────

-- Búsquedas por temperatura (común en el dashboard)
CREATE INDEX idx_experimentos_T ON experimentos(T);

-- Búsquedas por fuente de datos
CREATE INDEX idx_experimentos_fuente ON experimentos(fuente);

-- Búsquedas combinadas T + fuente (frecuentes)
CREATE INDEX idx_experimentos_T_fuente ON experimentos(T, fuente);

-- Joins desde mediciones hacia experimentos
CREATE INDEX idx_mediciones_experimento ON mediciones(id_experimento);

-- Para queries de series temporales del lab UdeC
CREATE INDEX idx_mediciones_timestamp 
    ON mediciones(timestamp_medicion) 
    WHERE timestamp_medicion IS NOT NULL;

-- ───────────────────────────────────────────────────────────────
-- TRIGGER: Actualización automática de updated_at
-- ───────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION trigger_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_updated_at_experimentos
    BEFORE UPDATE ON experimentos
    FOR EACH ROW
    EXECUTE FUNCTION trigger_set_updated_at();

CREATE TRIGGER set_updated_at_parametros_modelo
    BEFORE UPDATE ON parametros_modelo
    FOR EACH ROW
    EXECUTE FUNCTION trigger_set_updated_at();

-- ───────────────────────────────────────────────────────────────
-- VISTA: experimentos_completos
-- Combina condiciones + parámetros del modelo (uso frecuente en dashboard)
-- ───────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW experimentos_completos AS
SELECT 
    e.*,
    p.E_max,
    p.i_max,
    p.r_1,
    p.r_2,
    p.n_H2_a_in,
    (SELECT COUNT(*) FROM mediciones m 
     WHERE m.id_experimento = e.id_experimento) AS n_mediciones
FROM experimentos e
LEFT JOIN parametros_modelo p ON e.id_experimento = p.id_experimento;

COMMENT ON VIEW experimentos_completos IS 
    'Vista consolidada: condiciones experimentales + parámetros del modelo + conteo de mediciones';