# Estructura sugerida del reporte

Ajústenla a lo que pida el profesor (rúbrica, formato, extensión).

1. **Portada** — título, integrantes (11), grupo, materia, fecha.
2. **Introducción** — problema, objetivo, alcance. *(Persona 1)*
3. **Marco teórico** *(Persona 1)*
   - Detección de rostros (Haar / MediaPipe).
   - Características HOG y clasificadores clásicos (SVM, KNN, Random Forest).
   - Vision Transformer (ViT) y fine-tuning.
   - Métricas: accuracy, precisión, recall, F1, matriz de confusión.
4. **Dataset y preprocesamiento** — personas, fotos por persona, cómo se tomaron, recorte, tamaño, partición train/val/test y cómo se evitó la fuga de datos. *(Preprocesamiento, persona 4)*
5. **Metodología** — descripción de cada modelo e hiperparámetros, entorno (Colab/GPU), semilla, criterio para elegir el mejor modelo. *(Persona 1 con datos de Entrenamiento)*
6. **Resultados** — tabla comparativa (`resultados/comparacion.md`), gráfica, matrices de confusión, métricas por persona. *(Persona 2)*
7. **Discusión** — qué modelo funcionó mejor y por qué, errores típicos (fotos mal clasificadas), limitaciones del dataset (iluminación, ángulo, pocas fotos). *(Persona 2)*
8. **Demo** — capturas de la demo en vivo y cómo se usa. *(Persona 3)*
9. **Conclusiones y trabajo futuro** *(Persona 2)*
10. **Referencias** *(Persona 1; revisión final Persona 3)*
11. **Anexos** — enlace al repositorio, reparto de tareas por equipo.

## Lista de revisión antes de entregar
- [ ] Los números del texto coinciden con `resultados/`.
- [ ] Toda figura y tabla está numerada y mencionada en el texto.
- [ ] Se explica cómo se evitó evaluar en datos vistos (test usado una sola vez).
- [ ] Ortografía y formato revisados por alguien que no escribió esa sección.
- [ ] Nombres de todos los integrantes y reparto de tareas incluidos.
