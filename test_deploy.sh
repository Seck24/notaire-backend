#!/bin/bash
# Test post-déploiement — Notaire Agentia Backend
# Usage: ./test_deploy.sh [BASE_URL]
# Par défaut: https://api.notaire.preo-ia.info

BASE_URL="${1:-https://api.notaire.preo-ia.info}"
FRONTEND_ORIGIN="https://notaire-agentia.preo-ia.info"
PASS=0
FAIL=0

ok()   { echo "✅ $1"; ((PASS++)); }
fail() { echo "❌ $1"; ((FAIL++)); }

echo "=== Tests post-déploiement — $BASE_URL ==="
echo ""

# Test 1 — Health check
echo "Test 1 — GET /health"
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
if [ "$HEALTH" = "200" ]; then
  ok "GET /health → 200"
else
  fail "GET /health → $HEALTH (attendu 200)"
fi

# Test 2 — CORS preflight
echo ""
echo "Test 2 — OPTIONS CORS /api/generer-acte"
CORS=$(curl -s -o /dev/null -w "%{http_code}" -X OPTIONS "$BASE_URL/api/generer-acte" \
  -H "Origin: $FRONTEND_ORIGIN" \
  -H "Access-Control-Request-Method: POST")
if [ "$CORS" = "200" ] || [ "$CORS" = "204" ]; then
  ok "OPTIONS CORS → $CORS"
else
  fail "OPTIONS CORS → $CORS (attendu 200 ou 204)"
fi

# Test 3 — verify-token (token invalide → réponse JSON attendue)
echo ""
echo "Test 3 — POST /api/verify-token"
TOKEN_RESP=$(curl -s -X POST "$BASE_URL/api/verify-token" \
  -H "Content-Type: application/json" \
  -d '{"token": "test_invalide", "cabinet_id": "test"}')
if echo "$TOKEN_RESP" | grep -q '"valid"'; then
  ok "POST /api/verify-token → JSON avec champ 'valid'"
else
  fail "POST /api/verify-token → réponse inattendue: $TOKEN_RESP"
fi

# Résumé
echo ""
echo "=== Résumé : $PASS OK / $((PASS+FAIL)) tests ==="
if [ $FAIL -eq 0 ]; then
  echo "✅ Tous les tests passent — backend opérationnel"
else
  echo "⚠️  $FAIL test(s) en échec — vérifier les logs Coolify"
fi
