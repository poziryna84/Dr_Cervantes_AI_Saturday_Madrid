# Dr_Cervantes_AI_Saturday_Madrid
Proyecto de NLP de clasificación de temas (categorías clínicas).

Datos: contienen sintomas de los pacientes en el momento de ingreso en hospital (columna 'MOTIVO_CONSULTA') y el diagnostico del doctor
(la columna 'GENERAL_DIAG').

Ficheros:
Unificacion_tablones_medical_data.ipynb: los cruces de las tablas con los datos en crudo con los codigos de diagnosticos estandarizados.
Limpieza de los registros duplicados y aquellos que no disponen el diagnostico final.

Sintomas_medicos_modelado.ipynb: preprocemiento de los textos de sintomas y modelado final.

El objetivo del proyecto es crear un modelo de clasificacion multiple para predicir el diagnostico medico segun los sintomas del paciente.
Para implementar el modelo se crea una api usando flask.
