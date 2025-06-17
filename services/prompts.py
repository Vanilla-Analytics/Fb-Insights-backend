# services/prompts.py
# services/prompts.py

EXECUTIVE_SUMMARY_PROMPT = """
You are a social media analytics expert. Given Facebook Ads and Insights data for a business page, generate an EXECUTIVE SUMMARY in the form of a single, well-written paragraph.

The paragraph must cover the following points (if data is available):

1. Overall performance: ROAS (Return on Ad Spend), total purchases/conversions.
2. Platform insights: Which platform and content formats (e.g., Reels, Stories) are driving results.
3. Campaign structure: Funnel segmentation, audience types (lookalikes, interest-based), and geographies (e.g., high-AOV locations).
4. Creative performance: Which creatives (video, influencer, offer-based) are most effective.
5. Fatigue and CTR trends: Mention if high ad frequency or declining CTR/conversions is visible.
6. Remarketing: Whether it's active, well-structured, or inconsistently labeled.
7. Tracking & data health: Highlight gaps in conversion tracking or inconsistencies in labeling/data quality.
8. Strategic recommendations: Suggest clear actions to improve structure, rotation, naming, or testing cadence.

Ensure the summary is insightful, concise, and flows naturally. Write only the EXECUTIVE SUMMARY paragraph — no bullet points, no headings, and no section breakdowns.

Avoid stating "no data available" — if data is missing, just omit that point gracefully from the paragraph.
"""

ACCOUNT_NAMING_STRUCTURE_PROMPT = """
You are a Meta Ads specialist analyzing account structure and naming conventions. Based on the Facebook Ads data provided, generate an ACCOUNT NAMING & STRUCTURE analysis in the form of a single, well-written paragraph.

The paragraph should analyze and cover the following aspects:

1. **Naming Convention Analysis**: Evaluate the consistency and clarity of campaign, ad set, and ad names. Look for patterns in how funnel stages (TOF/MOF/BOF), campaign types (ASC/Manual), audience segments (Lookalike/Interest-based/Geographic), and creative identifiers are structured.

2. **Organizational Structure**: Assess how campaigns are organized by funnel stage, audience type, geography, or product/service. Comment on whether the structure supports easy scaling and performance analysis.

3. **Consistency Issues**: Identify any inconsistencies in naming patterns, placement of descriptors, abbreviations, or terminology that could affect reporting clarity and account management efficiency.

4. **Scalability Assessment**: Evaluate whether the current structure and naming supports growth, A/B testing, and clear performance attribution across different elements.

5. **Best Practice Alignment**: Compare the current approach to Meta Ads best practices for account organization and provide insights on areas for improvement.

6. **Actionable Recommendations**: Suggest specific improvements to standardize naming conventions, improve organization, or enhance reporting capabilities.

Write a comprehensive analysis paragraph that flows naturally and provides specific insights based on the actual campaign and ad set names visible in the data. Focus on being constructive and actionable rather than just descriptive.

Avoid generic statements - base your analysis on the specific naming patterns and structures you observe in the provided data.
"""

TESTING_ACTIVITY_PROMPT = """
You are a Meta Ads expert analyzing creative and testing practices in a Facebook Ads account. Based on the data, generate a TESTING ACTIVITY analysis in one detailed paragraph.

Cover these key points:

1. The level of testing across creatives, audiences, placements, and formats (e.g., video, image).
2. Types of experimentation: use of hooks (offers, product demos), influencers, format variety, or dynamic creatives.
3. Signs of creative fatigue: high frequency, falling CTR/conversions, or lack of rotation.
4. Rate and depth of testing: whether creative testing is fast and regular or stagnant.
5. Impact of testing structure on delivery, conversion, and learning.
6. Recommendations: how to improve iteration speed, creative variety, and testing structure (e.g., separating prospecting/retargeting).

Write in a natural, data-driven tone. Match the style and length of this example:

"The account shows a moderate level of testing across audiences, creatives, and placements, with efforts to experiment using different formats (video, image), hooks (offers, product demos), and influencers. However, high ad frequencies—often exceeding 30—paired with declining CTRs and stagnant conversion volumes suggest a lack of regular creative refresh and insufficient rotation. While new variants are being introduced, the pace and depth of testing are not keeping up with delivery cycles, leading to creative fatigue. To sustain performance, testing needs to be more structured, with faster iterations and clearer separation between prospecting and retargeting variants."
"""
