#!/bin/bash

echo "🔄 GraphRAG Restart Progress Check"
echo "================================="

# Check if processes are running
RUNNING=$(ps aux | grep "graphrag index" | grep -v grep | wc -l)
echo "📊 Active processes: $RUNNING"

if [ $RUNNING -gt 0 ]; then
    echo "✅ GraphRAG restart is running..."
else
    echo "🏁 GraphRAG restart has finished!"
fi

echo ""
echo "📁 Output files created:"
ls -la data/output/ | grep -v "^d" | grep -v "^total"

echo ""
echo "💾 Cache files available: $(find data/cache/ -type f | wc -l)"

echo ""
echo "📈 Current restart log tail:"
tail -5 graphrag_restart.log 2>/dev/null || echo "No restart log yet"

echo ""
echo "⏰ Last updated: $(date)"
echo ""
echo "💡 Run this script again: ./check_restart.sh" 