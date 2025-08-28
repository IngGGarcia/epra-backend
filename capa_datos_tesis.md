# Definición de la Capa de Datos - Sistema EPRA

## Resumen Ejecutivo

El sistema EPRA (EEG Processing and Response Analysis) constituye una plataforma integral para el procesamiento y análisis de señales electroencefalográficas (EEG) en estudios de evaluación emocional mediante estímulos visuales. La capa de datos del sistema está diseñada siguiendo principios de modularidad, responsabilidad atómica y referencias relacionales débilmente acopladas, evitando intencionalmente las restricciones de claves foráneas explícitas para promover la flexibilidad y mantener la consistencia lógica a nivel de aplicación.

## Arquitectura General del Sistema

### Propósito y Contexto

El sistema EPRA está diseñado para:
- **Clasificación automática pre-experimental**: Modelo de ML entrenado que categoriza imágenes según su nivel de violencia (escala 0-4)
- **Gestión de evaluaciones EEG multiusuario**: Sesiones experimentales controladas con múltiples participantes
- **Procesamiento de señales electroencefalográficas**: Análisis en tiempo real de respuestas neurológicas a estímulos visuales
- **Cálculo científicamente validado**: Valencia y activación emocional basados en modelos neurocientíficos establecidos
- **Validación cruzada SAM-EEG**: Comparación entre métricas subjetivas y objetivas para validación de modelos

### Fundamentos Científicos

#### Modelo de Clasificación de Violencia (Componente Primario)

El sistema inicia con un **modelo de machine learning pre-entrenado** que clasifica automáticamente las imágenes según su contenido violento:

- **Entrada**: Imágenes digitales para análisis de contenido
- **Clasificación binaria**: Determinación inicial violenta/no violenta (`is_violent_original`)
- **Clasificación graduada**: Escala refinada de 0-4 niveles de violencia (`violent_final_classification`)
- **Vector de características**: Representación numérica multidimensional utilizada para la decisión del modelo
- **Propósito**: Preparar estímulos calibrados para experimentos EEG controlados

#### Algoritmos EEG Científicamente Validados

El sistema implementa métodos neurocientíficos establecidos para análisis de respuestas emocionales:
- **Modelo de Asimetría Frontal Alpha de Davidson** para cálculo de valencia emocional
- **Ratio Beta/Alpha** para medición de activación cortical  
- **Sistema Internacional 10-20** para posicionamiento estandarizado de electrodos
- **Estándares internacionales** para definición de bandas de frecuencia EEG

## Descripción Detallada de las Tablas

### 1. Tabla `evaluation`

#### Propósito
Esta tabla representa las sesiones de evaluación principales, actuando como el contenedor de alto nivel para las evaluaciones EEG individuales realizadas por cada usuario.

#### Estructura de Datos

```sql
CREATE TABLE evaluation (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    eeg_file_path TEXT,
    sam_file_path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_evaluation_user_id ON evaluation(user_id);
```

#### Definición de Campos

| Campo | Tipo | Descripción | Restricciones |
|-------|------|-------------|---------------|
| `id` | TEXT | Identificador único generado automáticamente (UUID4) | PRIMARY KEY |
| `user_id` | INTEGER | Identificador del usuario asociado a la evaluación | NOT NULL, INDEX |
| `eeg_file_path` | TEXT | Ruta del archivo EEG sin procesar cargado por el usuario | NULLABLE |
| `sam_file_path` | TEXT | Ruta del archivo de respuestas SAM (Self-Assessment Manikin) | NULLABLE |
| `created_at` | DATETIME | Timestamp de creación del registro | DEFAULT CURRENT_TIMESTAMP |
| `updated_at` | DATETIME | Timestamp de última actualización | DEFAULT CURRENT_TIMESTAMP |

#### Consideraciones de Diseño

- **Identificador UUID**: Garantiza unicidad global y evita colisiones en sistemas distribuidos
- **Índice en user_id**: Optimiza consultas frecuentes de evaluaciones por usuario
- **Rutas de archivos opcionales**: Permite evaluaciones en progreso sin archivos completos
- **Timestamps automáticos**: Facilita auditoría y trazabilidad temporal

### 2. Tabla `image_classification` (Tabla Primaria del Sistema)

#### Propósito
Esta tabla constituye el punto de partida del sistema EPRA, almacenando los resultados del modelo de clasificación de violencia entrenado. Cada registro representa una imagen procesada por el modelo de machine learning que determina el nivel de violencia visual en una escala graduada de 0 a 4, junto con la clasificación binaria original y el vector de características utilizado para la toma de decisión.

#### Estructura de Datos

```sql
CREATE TABLE image_classification (
    image_id TEXT PRIMARY KEY,
    is_violent_original BOOLEAN NOT NULL,
    violent_final_classification INTEGER NOT NULL,
    feature_vector TEXT DEFAULT ''
);
```

#### Definición de Campos

| Campo | Tipo | Descripción | Restricciones |
|-------|------|-------------|---------------|
| `image_id` | TEXT | Identificador único de la imagen | PRIMARY KEY, INDEX |
| `is_violent_original` | BOOLEAN | Clasificación binaria original (violenta/no violenta) de la imagen | NOT NULL |
| `violent_final_classification` | INTEGER | Nivel de violencia graduado [0-4] determinado por el modelo de ML | NOT NULL, RANGE [0,4] |
| `feature_vector` | TEXT | Vector de características extraído de la imagen para decisión del modelo | DEFAULT '' |

#### Escala de Clasificación de Violencia

| Nivel | Descripción | Características |
|-------|-------------|-----------------|
| 0 | No violenta | Contenido pacífico, sin elementos agresivos |
| 1 | Violencia mínima | Indicios sutiles, tensión implícita |
| 2 | Violencia moderada | Elementos agresivos evidentes pero controlados |
| 3 | Violencia alta | Contenido explícitamente violento |
| 4 | Violencia extrema | Máximo nivel de contenido violento |

#### Consideraciones de Diseño

- **Tabla fundacional**: Primera en ser poblada, base para evaluaciones posteriores
- **Clasificación graduada**: Escala 0-4 permite análisis más granular que clasificación binaria
- **Vector de características**: Preserva la evidencia utilizada por el modelo para reproducibilidad
- **Trazabilidad del modelo**: Mantiene tanto clasificación original como refinada para validación
- **Serialización JSON**: Flexibilidad para vectores de dimensiones variables según arquitectura del modelo

### 3. Tabla `image_evaluation`

#### Propósito
Esta tabla constituye el núcleo del sistema, registrando las evaluaciones detalladas a nivel de imagen individual. Cada registro captura la asociación entre un usuario específico, un estímulo visual particular y las respuestas tanto subjetivas (escala SAM) como objetivas (métricas EEG calculadas).

#### Estructura de Datos

```sql
CREATE TABLE image_evaluation (
    id TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    sam_valence INTEGER NOT NULL,
    sam_arousal INTEGER NOT NULL,
    eeg_valence REAL NOT NULL,
    eeg_arousal REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_image_evaluation_user_id ON image_evaluation(user_id);
```

#### Definición de Campos

| Campo | Tipo | Descripción | Restricciones |
|-------|------|-------------|---------------|
| `id` | TEXT | Identificador único del registro de evaluación de imagen (UUID4) | PRIMARY KEY |
| `evaluation_id` | TEXT | Referencia lógica a la sesión de evaluación principal | NOT NULL |
| `user_id` | INTEGER | Identificador del usuario asociado | NOT NULL, INDEX |
| `image_id` | INTEGER | Identificador del estímulo visual presentado | NOT NULL |
| `sam_valence` | INTEGER | Puntuación de valencia según escala SAM [1-9] | NOT NULL |
| `sam_arousal` | INTEGER | Puntuación de activación según escala SAM [1-9] | NOT NULL |
| `eeg_valence` | REAL | Valencia calculada automáticamente desde señales EEG | NOT NULL |
| `eeg_arousal` | REAL | Activación calculada automáticamente desde señales EEG | NOT NULL |
| `created_at` | TIMESTAMP | Timestamp de creación del registro | DEFAULT CURRENT_TIMESTAMP |
| `updated_at` | TIMESTAMP | Timestamp de última actualización | DEFAULT CURRENT_TIMESTAMP |

#### Consideraciones de Diseño

- **Evaluación dual**: Combina métricas subjetivas (SAM) y objetivas (EEG)
- **Granularidad por imagen**: Permite análisis detallado de respuestas específicas
- **Escalas normalizadas**: SAM utiliza escala estándar 1-9 para comparabilidad
- **Precisión EEG**: Valores flotantes para métricas neurofisiológicas precisas

## Fundamentos Científicos de los Cálculos EEG

### Cálculo de Valencia (Modelo de Davidson)

**Fórmula**: `valencia = log(potencia_alpha_derecha) - log(potencia_alpha_izquierda)`

**Principio neurofisiológico**:
- Activación frontal izquierda (menor alpha) → valencia positiva/motivación de acercamiento
- Activación frontal derecha (menor alpha) → valencia negativa/motivación de retirada
- La supresión de potencia alpha indica activación cortical

**Validación científica**:
- Davidson, R.J. (2004): "What does the prefrontal cortex 'do' in affect"
- Wheeler, R.E., Davidson, R.J., & Tomarken, A.J. (1993): "Frontal brain asymmetry and emotional reactivity"
- Precisión validada: 70-85% en múltiples estudios

### Cálculo de Activación (Ratio Beta/Alpha)

**Fórmula**: `activacion = potencia_beta / potencia_alpha`

**Principio neurofisiológico**:
- Alta actividad beta → mayor activación cortical, atención, procesamiento cognitivo
- Baja actividad alpha → corteza desincronizada, procesamiento activo de información
- El ratio proporciona medida robusta de activación independiente de diferencias individuales

**Validación científica**:
- PMC11445052 (2024): 72% de precisión para evaluación de consciencia
- Applied Sciences 15(9):4980 (2025): Mejora del 85% al 95% en precisión de detección de activación
- Klimesch, W. (1999): "EEG alpha and theta oscillations reflect cognitive and memory performance"

### Definiciones de Bandas de Frecuencia

Según estándares internacionales del Sistema 10-20:

| Banda | Frecuencia | Función |
|-------|------------|---------|
| Delta | 1-4 Hz | Sueño profundo, procesos inconscientes |
| Theta | 4-8 Hz | Somnolencia, meditación, memoria |
| Alpha | 8-13 Hz | Conciencia relajada, ojos cerrados |
| Beta | 13-30 Hz | Pensamiento activo, resolución de problemas, atención |
| Gamma | 30-50 Hz | Binding, consciencia, funciones cognitivas de alto nivel |

### Selección de Electrodos

**Electrodos frontales**: AF3, AF4, F3, F4 (Sistema Internacional 10-20)

**Justificación científica**:
- Pizzagalli, D.A. (2007): Localización de fuentes electrofisiológicas de alta densidad
- Jasper, H.H. (1958): Sistema de electrodos ten-twenty de la Federación Internacional
- Keil, A., et al. (2014): Guías de publicación para estudios de electroencefalografía

## Relaciones entre Tablas

### Modelo de Referencia Lógica

El diseño implementa un modelo de referencia lógica sin restricciones de claves foráneas explícitas, con `image_classification` como tabla fundacional:

```
image_classification (1) ←→ (N) image_evaluation
         ↑                        ↓
    image_id                  evaluation_id
         ↑                        ↓
    [Modelo ML]            evaluation (1)
                                  ↓
                              user_id
```

#### Flujo de Datos (Secuencial)

1. **Pre-procesamiento con ML**: Las imágenes se procesan con el modelo de machine learning entrenado y se clasifican según su nivel de violencia (0-4), poblando `image_classification` con:
   - Clasificación binaria original (`is_violent_original`)
   - Nivel de violencia graduado (`violent_final_classification`)
   - Vector de características utilizado para la decisión (`feature_vector`)

2. **Creación de sesión de evaluación**: Se registra una nueva sesión experimental en `evaluation` para un usuario específico, incluyendo rutas de archivos EEG y SAM

3. **Evaluación EEG por imagen**: Para cada imagen pre-clasificada, se registra la evaluación individual en `image_evaluation`, combinando:
   - Referencia a la sesión (`evaluation_id`)
   - Referencia a la imagen clasificada (`image_id`)
   - Métricas subjetivas SAM del usuario
   - Métricas objetivas calculadas desde señales EEG

#### Ventajas del Diseño

- **Flexibilidad**: Permite evolución del esquema sin restricciones rígidas
- **Escalabilidad**: Facilita distribución y particionamiento de datos
- **Mantenibilidad**: Consistencia controlada a nivel de aplicación
- **Robustez**: Menor acoplamiento entre componentes del sistema

## Consideraciones de Rendimiento

### Índices Implementados

1. **idx_evaluation_user_id**: Optimiza consultas de evaluaciones por usuario
2. **idx_image_evaluation_user_id**: Acelera agregaciones por usuario
3. **image_id como PRIMARY KEY**: Búsquedas directas O(1) en clasificaciones

### Estrategias de Optimización

- **Timestamps automáticos**: Evita lógica de aplicación redundante
- **UUIDs como TEXT**: Balance entre unicidad y rendimiento de consulta
- **Índices selectivos**: Solo en campos de consulta frecuente
- **Normalización controlada**: Evita joins complejos manteniendo consistencia

## Casos de Uso Principales

### 1. Pre-clasificación de Imágenes con Modelo ML (Paso Inicial)
```sql
-- Resultado del procesamiento del modelo de clasificación de violencia
INSERT INTO image_classification (
    image_id, is_violent_original, violent_final_classification, feature_vector
) VALUES (
    'image_001.jpg', true, 3, '[0.234, 0.567, 0.891, ..., 0.123]'
),
(
    'image_002.jpg', false, 0, '[0.123, 0.456, 0.789, ..., 0.012]'
),
(
    'image_003.jpg', true, 4, '[0.345, 0.678, 0.912, ..., 0.234]'
);
```

### 2. Creación de Nueva Evaluación (Sesión Experimental)
```sql
INSERT INTO evaluation (id, user_id, eeg_file_path, sam_file_path)
VALUES (uuid4(), 12345, '/data/eeg/session_001.csv', '/data/sam/responses_001.json');
```

### 3. Registro de Evaluación por Imagen (Datos Experimentales)
```sql
INSERT INTO image_evaluation (
    id, evaluation_id, user_id, image_id,
    sam_valence, sam_arousal, eeg_valence, eeg_arousal
) VALUES (
    uuid4(), 'eval_123', 12345, 'image_001.jpg',
    3, 8, -0.45, 7.23  -- Imagen violenta: baja valencia SAM, alta activación
),
(
    uuid4(), 'eval_123', 12345, 'image_002.jpg', 
    7, 2, 0.67, 1.89   -- Imagen no violenta: alta valencia SAM, baja activación
);
```

### 4. Análisis de Correlación entre Nivel de Violencia y Respuesta Emocional
```sql
SELECT 
    ic.violent_final_classification as nivel_violencia,
    AVG(ie.sam_valence) as valencia_sam_promedio,
    AVG(ie.eeg_valence) as valencia_eeg_promedio,
    AVG(ie.sam_arousal) as activacion_sam_promedio,
    AVG(ie.eeg_arousal) as activacion_eeg_promedio,
    COUNT(*) as total_evaluaciones
FROM image_evaluation ie
JOIN image_classification ic ON ie.image_id = ic.image_id
GROUP BY ic.violent_final_classification
ORDER BY ic.violent_final_classification;
```

### 5. Validación de Modelo: Comparación SAM vs EEG por Nivel de Violencia
```sql
SELECT 
    ic.violent_final_classification,
    AVG(ie.sam_valence - ie.eeg_valence) as diferencia_valencia,
    AVG(ie.sam_arousal - ie.eeg_arousal) as diferencia_activacion,
    STDDEV(ie.sam_valence - ie.eeg_valence) as desviacion_valencia,
    STDDEV(ie.sam_arousal - ie.eeg_arousal) as desviacion_activacion
FROM image_evaluation ie
JOIN image_classification ic ON ie.image_id = ic.image_id
GROUP BY ic.violent_final_classification
ORDER BY ic.violent_final_classification;
```

### 6. Consulta de Trazabilidad: Vector de Características del Modelo
```sql
SELECT 
    ic.image_id,
    ic.is_violent_original,
    ic.violent_final_classification,
    ic.feature_vector,
    AVG(ie.eeg_valence) as valencia_eeg_experimental,
    AVG(ie.eeg_arousal) as activacion_eeg_experimental
FROM image_classification ic
LEFT JOIN image_evaluation ie ON ic.image_id = ie.image_id
GROUP BY ic.image_id, ic.is_violent_original, ic.violent_final_classification, ic.feature_vector
ORDER BY ic.violent_final_classification DESC;
```

## Validación y Consistencia de Datos

### Reglas de Negocio Implementadas

1. **Validación temporal**: `created_at` ≤ `updated_at`
2. **Rangos SAM**: Valores entre 1-9 para valencia y activación
3. **Integridad referencial**: Validación de existencia a nivel de aplicación
4. **Unicidad de evaluaciones**: Una evaluación por imagen por usuario por sesión

### Mecanismos de Validación

- **Validación en modelo**: Restricciones de tipo y rango en SQLModel
- **Validación en servicio**: Lógica de negocio en capa de aplicación
- **Logging extensivo**: Trazabilidad completa de operaciones de datos
- **Manejo de errores**: Recuperación elegante de fallos de consistencia

## Escalabilidad y Futuras Extensiones

### Capacidad de Crecimiento

- **Particionamiento temporal**: Por fecha de creación para archivado
- **Sharding por usuario**: Distribución horizontal para gran escala
- **Índices compuestos**: Optimización de consultas complejas futuras
- **Materialización de vistas**: Pre-cálculo de agregaciones frecuentes

### Extensibilidad del Esquema

- **Campos adicionales**: Estructura flexible permite nuevas métricas
- **Nuevas tablas**: Diseño modular facilita componentes adicionales
- **Versioning de datos**: Soporte para evolución de algoritmos
- **Metadatos enriquecidos**: Capacidad para contexto experimental expandido

## Conclusiones

La capa de datos del sistema EPRA representa una arquitectura robusta y científicamente fundamentada para la gestión de evaluaciones EEG multi-dimensionales. El diseño prioriza la flexibilidad operacional, la validación científica y la escalabilidad futura, proporcionando una base sólida para investigación neurocientífica avanzada y aplicaciones clínicas.

Las decisiones de diseño reflejan un balance cuidadoso entre rigor científico, eficiencia operacional y adaptabilidad tecnológica, estableciendo un estándar para sistemas de procesamiento EEG en entornos de investigación doctoral y aplicaciones de producción.

---

*Este documento forma parte de la documentación técnica para tesis doctoral sobre procesamiento y análisis de señales EEG en evaluación emocional mediante estímulos visuales.*
