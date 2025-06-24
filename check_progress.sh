#!/bin/bash

echo "🔍 GraphRAG Indexing Progress Check"
echo "=================================="

# Check if processes are running
RUNNING=$(ps aux | grep "graphrag index" | grep -v grep | wc -l)
echo "📊 Active processes: $RUNNING"

if [ $RUNNING -gt 0 ]; then
    echo "✅ GraphRAG is still running..."
else
    echo "🏁 GraphRAG processes have finished!"
fi

echo ""
echo "📁 Output files created:"
ls -la data/output/ | grep -v "^d" | grep -v "^total"

echo ""
echo "📈 Current log tail:"
tail -5 graphrag_index.log

echo ""
echo "⏰ Last updated: $(date)"
echo ""
echo "💡 Run this script again to check progress: ./check_progress.sh" 