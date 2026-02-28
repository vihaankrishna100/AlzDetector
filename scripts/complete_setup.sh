
set -e  

echo "ADgent Framework - Complete Setup"
echo "======================================"
echo ""

echo "Step 1: Checking system readiness..."
python -m adgent.validate_system || {
    echo "Some checks failed - see above"
    echo "Most common issues:"
    echo "  • No model.pth → Run 'python scripts/train.py' below"
    echo "  • No LLM API key → Set OPENAI_API_KEY environment variable"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
}
echo "System check complete"
echo ""

if [ ! -f "model.pth" ]; then
    echo "Step 2: Training model..."
    echo "   (This will take 5-15 minutes)"
    python scripts/train.py
    echo "Model training complete"
else
    echo "Step 2: Skipping - model.pth already exists"
fi
echo ""

echo "Step 3: Checking LLM API key..."
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "No LLM API key found!"
    echo ""
    echo "Get one from:"
    echo "  • OpenAI: https://platform.openai.com/api-keys"
    echo "  • Anthropic: https://console.anthropic.com/"
    echo ""
    echo "Then set it with:"
    echo "  export OPENAI_API_KEY='sk-...'"
    echo ""
    read -p "Have you set the API key? (y/n) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
else
    echo "LLM API key found"
fi
echo ""

echo "Step 4: Configuring agents..."
if [ -f "agents/agents_fixed.py" ]; then
    cp agents/agents_fixed.py agents/agents.py
    echo "Agents configured with LLM"
else
    echo "agents/agents_fixed.py not found - skipping"
fi
echo ""

echo "Step 5: Testing on single subject..."
echo "   (This will take a minute or two)"
python -m agents.main
echo "Single subject test complete"
echo ""

echo "All done. Framework is ready."
echo "Next: Run 'python scripts/infer_agents.py' to test on 20 subjects"
echo ""
