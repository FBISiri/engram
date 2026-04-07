// Package reflection provides optional background reflection generation.
package reflection

// TODO: Implement reflection engine
//
// Triggers: count-based, cron-based, or manual
// Process:
// 1. Fetch recent un-reflected memories
// 2. Send to LLM with reflection prompt
// 3. Store result as type=insight, source=system
// 4. Mark source memories as reflected
