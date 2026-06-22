#!/bin/bash
# =============================================================================
# AI Comment Moderator — End-to-End Demo
# =============================================================================
# Runs the full moderation flow against a locally running server.
#
# Prerequisites:
#   1. Server is running: uvicorn main:app --reload
#   2. jq is installed: brew install jq
#
# Usage:
#   chmod +x demo.sh
#   ./demo.sh
# =============================================================================

BASE_URL="http://localhost:8000"
DIVIDER="─────────────────────────────────────────────────────"

# Colours
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

check_deps() {
  if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is required. Install with: brew install jq${RESET}"
    exit 1
  fi
  if ! curl -s "$BASE_URL/health" > /dev/null; then
    echo -e "${RED}Error: Server not running at $BASE_URL${RESET}"
    echo "Start it with: uvicorn main:app --reload"
    exit 1
  fi
}

print_step() {
  echo ""
  echo -e "${CYAN}${DIVIDER}${RESET}"
  echo -e "${BOLD}$1${RESET}"
  echo -e "${CYAN}${DIVIDER}${RESET}"
}

print_response() {
  echo "$1" | jq .
}

check_deps

echo ""
echo -e "${BOLD}🏠  PropertyTribes — AI Comment Moderator Demo${RESET}"
echo -e "${CYAN}${DIVIDER}${RESET}"

# ---------------------------------------------------------------------------
# Step 1: Approve a legitimate comment
# ---------------------------------------------------------------------------
print_step "Step 1 — Submit a legitimate comment (expect: approved)"

RESPONSE_1=$(curl -s -X POST "$BASE_URL/moderate" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "landlord_dave",
    "comment": "Has anyone dealt with Article 4 directions for HMOs in Manchester? I have a 5-bed property and I am unsure whether I need an additional licence on top of mandatory HMO licensing."
  }')

print_response "$RESPONSE_1"
DECISION_1=$(echo "$RESPONSE_1" | jq -r '.decision')
echo -e "\n${GREEN}Decision: $DECISION_1${RESET}"

# ---------------------------------------------------------------------------
# Step 2: Reject a spam comment
# ---------------------------------------------------------------------------
print_step "Step 2 — Submit a spam comment (expect: rejected)"

RESPONSE_2=$(curl -s -X POST "$BASE_URL/moderate" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "spammer_99",
    "comment": "Check out my website for the BEST property leads in the UK! Limited time offer — DM me now for exclusive deals and guaranteed returns!"
  }')

print_response "$RESPONSE_2"
SPAM_ID=$(echo "$RESPONSE_2" | jq -r '.comment_id')
DECISION_2=$(echo "$RESPONSE_2" | jq -r '.decision')
echo -e "\n${RED}Decision: $DECISION_2${RESET}"

# ---------------------------------------------------------------------------
# Step 3: Appeal the rejected comment
# ---------------------------------------------------------------------------
print_step "Step 3 — Appeal the rejected comment with professional context"

RESPONSE_3=$(curl -s -X POST "$BASE_URL/appeal" \
  -H "Content-Type: application/json" \
  -d "{
    \"comment_id\": \"$SPAM_ID\",
    \"appeal_context\": \"I am a RICS-qualified surveyor with 15 years of experience helping landlords source investment properties. My comment was intended as a professional service introduction, not spam. I am happy to disclose my credentials and remove the urgency language.\"
  }")

print_response "$RESPONSE_3"
APPEAL_DECISION=$(echo "$RESPONSE_3" | jq -r '.appeal_decision')
echo -e "\n${YELLOW}Appeal decision: $APPEAL_DECISION${RESET}"

# ---------------------------------------------------------------------------
# Step 4: Submit a borderline comment likely to be flagged
# ---------------------------------------------------------------------------
print_step "Step 4 — Submit a borderline comment (expect: flagged_for_review)"

RESPONSE_4=$(curl -s -X POST "$BASE_URL/moderate" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "seminar_host",
    "comment": "I run property investment workshops in London — next one is next Saturday. We cover buy-to-let strategy, HMO setup, and tax efficiency. Visit my site for details."
  }')

print_response "$RESPONSE_4"
FLAGGED_ID=$(echo "$RESPONSE_4" | jq -r '.comment_id')
DECISION_4=$(echo "$RESPONSE_4" | jq -r '.decision')
echo -e "\n${YELLOW}Decision: $DECISION_4${RESET}"

# ---------------------------------------------------------------------------
# Step 5: Admin override the flagged comment
# ---------------------------------------------------------------------------
print_step "Step 5 — Admin overrides the flagged comment → approved"

RESPONSE_5=$(curl -s -X PATCH "$BASE_URL/log/$FLAGGED_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "approved",
    "note": "Verified organiser — event is legitimate. Approved for Property Seminars tribe."
  }')

print_response "$RESPONSE_5"
echo -e "\n${GREEN}Admin override applied${RESET}"

# ---------------------------------------------------------------------------
# Step 6: View the full log
# ---------------------------------------------------------------------------
print_step "Step 6 — View the moderation log (page 1)"

RESPONSE_6=$(curl -s "$BASE_URL/log?page=1&limit=10")
COUNT=$(echo "$RESPONSE_6" | jq 'length')
echo -e "Log contains ${BOLD}$COUNT entries${RESET}:"
echo "$RESPONSE_6" | jq '[.[] | {comment_id, decision, appealed, admin_overridden}]'

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${CYAN}${DIVIDER}${RESET}"
echo -e "${BOLD}Demo complete.${RESET}"
echo ""
echo -e "  Legitimate comment  → ${GREEN}$DECISION_1${RESET}"
echo -e "  Spam comment        → ${RED}$DECISION_2${RESET}"
echo -e "  Appeal outcome      → ${YELLOW}$APPEAL_DECISION${RESET}"
echo -e "  Borderline comment  → ${YELLOW}$DECISION_4${RESET} → admin override → ${GREEN}approved${RESET}"
echo ""
echo -e "Full log: ${CYAN}$BASE_URL/log${RESET}"
echo -e "API docs: ${CYAN}$BASE_URL/docs${RESET}"
echo -e "${CYAN}${DIVIDER}${RESET}"
