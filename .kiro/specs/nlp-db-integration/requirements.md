# NLP Database Integration Requirements

## Overview
Integrate the existing NLP sentiment analysis pipeline with the MQS database system to enable real-time sentiment scoring and storage for trading strategies.

## Current State Analysis

### Existing NLP Pipeline
- **Article Collection**: 24/7 scraping from multiple sources (FMP, Yahoo, Finviz, Alpha Vantage)
- **Storage**: Articles saved to CSV files in `NLP/articles/`
- **Model**: FinBERT-based sentiment analysis in Jupyter notebooks
- **Output**: Sentiment scores saved to CSV files in `NLP/sentiment_scores/`
- **Visualization**: Interactive plots showing sentiment vs. stock performance

### Database Schema Status
- **✅ Completed**: `news_sentiment` table has been created in the database
- **Schema**: 
  ```sql
  CREATE TABLE news_sentiment (
      id SERIAL PRIMARY KEY,
      ticker VARCHAR(10),
      article_url TEXT,
      published_at TIMESTAMP,
      sentiment_score FLOAT, -- Range: -1.0 to 1.0
      content_summary TEXT
  );
  ```
- **Current Tables**: market_data, trade_execution_logs, pnl_book, risk_book, news_sentiment, etc.
- **Integration Point**: Need to connect NLP pipeline to populate the news_sentiment table

## User Stories

### US1: Database Integration Validation
**As a** system administrator  
**I want** to validate the existing `news_sentiment` table schema  
**So that** it meets the requirements for sentiment data storage  

**Acceptance Criteria:**
- 1.1 Verify `news_sentiment` table schema matches NLP pipeline data structure
- 1.2 Confirm table fields align with scraped article data (ticker, article_url, published_at, sentiment_score, content_summary)
- 1.3 Validate sentiment_score field supports FLOAT range -1.0 to 1.0
- 1.4 Test table supports efficient querying by ticker and published_at date range

### US2: Real-time Sentiment Processing
**As a** trading system  
**I want** newly scraped articles to be automatically processed for sentiment  
**So that** sentiment scores are available immediately for trading decisions  

**Acceptance Criteria:**
- 2.1 New articles trigger automatic sentiment analysis
- 2.2 FinBERT model processes article content and title
- 2.3 Sentiment scores are calculated within 30 seconds of article scraping
- 2.4 Processing handles batch operations for multiple articles efficiently

### US3: Database Integration Pipeline
**As a** trading system  
**I want** sentiment scores to be automatically saved to the database  
**So that** they can be accessed by trading strategies in real-time  

**Acceptance Criteria:**
- 3.1 Sentiment scores are inserted into `news_sentiment` table with all required fields (ticker, article_url, published_at, sentiment_score, content_summary)
- 3.2 Database operations handle connection failures gracefully with retry logic
- 3.3 Duplicate sentiment scores are prevented using article_url as unique identifier
- 3.4 Database writes are atomic and consistent
- 3.5 Content_summary field contains truncated article content for reference

### US4: Historical Data Migration
**As a** data analyst  
**I want** existing CSV sentiment data to be migrated to the database  
**So that** historical analysis can be performed using database queries  

**Acceptance Criteria:**
- 4.1 All existing CSV sentiment files are imported to database
- 4.2 Data integrity is maintained during migration
- 4.3 Migration process is idempotent and can be re-run safely
- 4.4 CSV files remain as backup after successful migration

### US5: Trading Strategy Integration
**As a** portfolio manager  
**I want** to query sentiment data from the database  
**So that** trading strategies can incorporate sentiment signals  

**Acceptance Criteria:**
- 5.1 Database provides efficient queries for sentiment by ticker and published_at date range
- 5.2 Sentiment data can be aggregated (daily averages, rolling windows) using published_at timestamps
- 5.3 API endpoints expose sentiment data to trading strategies
- 5.4 Query performance supports real-time trading requirements (<100ms)
- 5.5 Content_summary field enables article-level analysis and debugging

## Technical Requirements

### TR1: Database Schema Validation
- Validate existing `news_sentiment` table schema aligns with NLP pipeline data
- Map article data fields to database columns (publishedDate → published_at, site → article_url, etc.)
- Verify sentiment_score FLOAT field supports FinBERT output range (-1.0 to 1.0)
- Test query performance with sample data using ticker and published_at indexes

### TR2: Model Integration
- FinBERT model accessible from Python scripts (not just notebooks)
- Model loading optimized for production use (cached, GPU support)
- Batch processing capabilities for multiple articles
- Error handling for malformed article content

### TR3: Data Pipeline Architecture
- Event-driven processing triggered by new article scraping
- Asynchronous processing to avoid blocking article collection
- Monitoring and logging for sentiment processing pipeline
- Configuration management for model parameters and database connections

### TR4: Performance Requirements
- Process 100+ articles per minute during peak news periods
- Database writes complete within 5 seconds of sentiment calculation
- Support concurrent processing of multiple tickers
- Memory usage optimized for 24/7 operation

### TR5: Data Quality and Validation
- Sentiment scores validated to be within expected range (-1.0 to 1.0)
- Article content validation before processing
- Data mapping validation (CSV fields → database columns)
- Error logging and alerting for failed processing
- Content_summary field truncation to reasonable length for database storage

## Non-Functional Requirements

### NFR1: Reliability
- 99.9% uptime for sentiment processing pipeline
- Graceful degradation when model or database unavailable
- Automatic recovery from transient failures
- Data consistency maintained across system restarts

### NFR2: Scalability
- Support processing for 50+ tickers simultaneously
- Handle 10,000+ articles per day
- Database schema supports millions of sentiment records
- Horizontal scaling capability for increased load

### NFR3: Maintainability
- Clear separation between data collection, processing, and storage
- Configurable model parameters and database connections
- Comprehensive logging and monitoring
- Documentation for operational procedures

## Success Metrics

### SM1: Integration Success
- 100% of new articles processed for sentiment within 60 seconds
- Zero data loss during CSV to database migration
- All existing functionality preserved after integration

### SM2: Performance Metrics
- Average sentiment processing time < 10 seconds per article
- Database query response time < 100ms for typical trading queries
- System memory usage stable over 24+ hour periods

### SM3: Data Quality Metrics
- Sentiment score accuracy validated against manual samples
- Zero duplicate records in database
- 100% data lineage tracking for audit purposes

## Dependencies

### Internal Dependencies
- Existing NLP article scraping system (24/7 daemon)
- MQS database system and schema management
- FinBERT model and training pipeline
- Trading strategy framework for consumption

### External Dependencies
- PostgreSQL database system
- Python ML libraries (transformers, torch, pandas)
- GPU resources for model inference (optional but recommended)
- Monitoring and alerting infrastructure

## Risks and Mitigations

### Risk 1: Model Performance Impact
**Risk**: FinBERT processing may slow down article collection  
**Mitigation**: Implement asynchronous processing pipeline with queues

### Risk 2: Database Connection Issues
**Risk**: Database outages could cause sentiment data loss  
**Mitigation**: Implement local caching and retry mechanisms

### Risk 3: Data Migration Complexity
**Risk**: CSV to database migration may introduce data inconsistencies  
**Mitigation**: Comprehensive validation and rollback procedures

### Risk 4: Real-time Processing Bottlenecks
**Risk**: High-volume news periods may overwhelm processing capacity  
**Mitigation**: Implement priority queuing and load balancing

## Out of Scope

- Changes to existing article scraping logic or sources
- Modifications to FinBERT model architecture or training
- Real-time streaming infrastructure (initial implementation will be near real-time)
- Advanced sentiment analysis features (entity extraction, topic modeling)
- Integration with external sentiment data providers
- Historical backtesting framework modifications