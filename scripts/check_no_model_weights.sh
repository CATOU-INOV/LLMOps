#!/usr/bin/env bash
# Bloque le commit si des poids de modèle (.pt, .bin, .safetensors) sont en staging
if git diff --cached --name-only | grep -qE '\.(pt|bin|safetensors)$'; then
  echo "ERREUR : poids de modèle détectés dans le staging. Ajoutez-les au .gitignore."
  exit 1
fi
exit 0
