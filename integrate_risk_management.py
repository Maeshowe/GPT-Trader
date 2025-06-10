#!/usr/bin/env python3
"""
Risk Management Integration Script
Integrates risk assessment into the existing GPT-Trader pipeline
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from risk_management import RiskManager, assess_portfolio_risk

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def integrate_risk_into_generator(portfolio_file: Path = None) -> Dict:
    """
    Integrate risk assessment into portfolio generation process
    This modifies the portfolio weights based on risk assessment
    """
    
    if portfolio_file is None:
        portfolio_file = Path(__file__).parent / "outputs" / "portfolio_latest.json"
    
    logger.info("🛡️ Starting risk-adjusted portfolio generation")
    
    try:
        # 1. Load original portfolio
        with open(portfolio_file) as f:
            portfolio_data = json.load(f)
        
        original_table = portfolio_data['table'].copy()
        logger.info(f"📊 Loaded portfolio with {len(original_table)} assets")
        
        # 2. Perform comprehensive risk assessment
        logger.info("🔍 Performing comprehensive risk assessment...")
        risk_assessment = assess_portfolio_risk(portfolio_file)
        
        if 'error' in risk_assessment:
            logger.error(f"❌ Risk assessment failed: {risk_assessment['error']}")
            return portfolio_data  # Return original if risk assessment fails
        
        # 3. Apply risk-based weight adjustments
        logger.info("⚖️ Applying risk-based weight adjustments...")
        adjusted_table = apply_risk_adjustments(original_table, risk_assessment)
        
        # 4. Create risk-adjusted portfolio
        risk_adjusted_portfolio = {
            'date': portfolio_data['date'],
            'table': adjusted_table,
            'risk_assessment': {
                'assessment_timestamp': risk_assessment['assessment_timestamp'],
                'overall_summary': risk_assessment['overall_summary'],
                'adjustments_applied': True,
                'original_weights_backup': original_table
            }
        }
        
        # 5. Save risk-adjusted portfolio
        risk_adjusted_file = portfolio_file.parent / "portfolio_risk_adjusted.json"
        with open(risk_adjusted_file, 'w') as f:
            json.dump(risk_adjusted_portfolio, f, indent=2)
        
        logger.info(f"✅ Risk-adjusted portfolio saved to {risk_adjusted_file}")
        
        # 6. Generate risk report
        generate_risk_report(risk_assessment, portfolio_file.parent)
        
        return risk_adjusted_portfolio
        
    except Exception as e:
        logger.error(f"❌ Risk integration failed: {e}")
        return portfolio_data if 'portfolio_data' in locals() else {}

def apply_risk_adjustments(portfolio_table: List[Dict], risk_assessment: Dict) -> List[Dict]:
    """Apply risk-based adjustments to portfolio weights"""
    
    adjusted_table = []
    total_weight_reduction = 0
    governance_risks = risk_assessment.get('governance_risks', {})
    alerts = risk_assessment.get('alerts', [])
    
    # Risk adjustment factors
    GOVERNANCE_RISK_CUTS = {
        'LOW': 0.0,
        'MEDIUM': 0.05,    # 5% reduction
        'HIGH': 0.15,      # 15% reduction  
        'CRITICAL': 0.30   # 30% reduction
    }
    
    logger.info("📋 Applying risk adjustments:")
    
    for asset in portfolio_table:
        ticker = asset['Asset'].strip().upper()
        original_weight = float(asset['Weight (%)'])
        adjusted_weight = original_weight
        adjustments_applied = []
        
        # 1. Governance risk adjustments
        if ticker in governance_risks:
            gov_risk = governance_risks[ticker]
            risk_level = gov_risk.get('risk_level', 'MEDIUM')
            
            if risk_level in GOVERNANCE_RISK_CUTS:
                cut_factor = GOVERNANCE_RISK_CUTS[risk_level]
                weight_reduction = original_weight * cut_factor
                adjusted_weight -= weight_reduction
                total_weight_reduction += weight_reduction
                
                if cut_factor > 0:
                    adjustments_applied.append(f"Gov risk ({risk_level}): -{cut_factor*100:.0f}%")
                    logger.info(f"  🔻 {ticker}: Governance risk {risk_level} → -{cut_factor*100:.0f}% weight")
        
        # 2. Alert-based adjustments
        ticker_alerts = [alert for alert in alerts if alert.get('ticker') == ticker]
        for alert in ticker_alerts:
            if alert.get('level') == 'CRITICAL':
                critical_cut = original_weight * 0.20  # Additional 20% cut for critical alerts
                adjusted_weight -= critical_cut
                total_weight_reduction += critical_cut
                adjustments_applied.append("Critical alert: -20%")
                logger.warning(f"  🚨 {ticker}: Critical alert → additional -20% weight")
        
        # 3. Minimum weight floor (don't go below 0.5%)
        adjusted_weight = max(adjusted_weight, 0.5)
        
        # Update asset entry
        adjusted_asset = asset.copy()
        adjusted_asset['Weight (%)'] = round(adjusted_weight, 2)
        
        # Add risk information to asset
        if adjustments_applied:
            risk_note = "; ".join(adjustments_applied)
            if 'Risk' in adjusted_asset:
                adjusted_asset['Risk'] += f" | RISK ADJ: {risk_note}"
            else:
                adjusted_asset['Risk'] = f"RISK ADJ: {risk_note}"
        
        adjusted_table.append(adjusted_asset)
    
    # 4. Redistribute reduced weights proportionally to remaining assets
    if total_weight_reduction > 0:
        logger.info(f"📊 Redistributing {total_weight_reduction:.2f}% weight reduction")
        
        # Calculate redistribution weights (proportional to current weights)
        total_current_weight = sum(float(asset['Weight (%)']) for asset in adjusted_table)
        
        for asset in adjusted_table:
            current_weight = float(asset['Weight (%)'])
            if current_weight > 1.0:  # Only redistribute to meaningful positions
                proportion = current_weight / total_current_weight
                additional_weight = total_weight_reduction * proportion
                asset['Weight (%)'] = round(current_weight + additional_weight, 2)
    
    # 5. Final normalization to ensure weights sum to 100%
    total_final_weight = sum(float(asset['Weight (%)']) for asset in adjusted_table)
    if abs(total_final_weight - 100.0) > 0.1:
        logger.info(f"🔧 Final normalization: {total_final_weight:.2f}% → 100.0%")
        
        for asset in adjusted_table:
            asset['Weight (%)'] = round(float(asset['Weight (%)']) * 100.0 / total_final_weight, 2)
    
    logger.info("✅ Risk adjustments complete")
    return adjusted_table

def generate_risk_report(risk_assessment: Dict, output_dir: Path):
    """Generate comprehensive risk report"""
    
    logger.info("📄 Generating risk report...")
    
    try:
        report = {
            "executive_summary": generate_executive_summary(risk_assessment),
            "detailed_analysis": {
                "governance_risks": risk_assessment.get('governance_risks', {}),
                "portfolio_metrics": risk_assessment.get('portfolio_risk', {}).get('portfolio_metrics', {}),
                "concentration_analysis": risk_assessment.get('portfolio_risk', {}).get('concentration_risk', {}),
                "liquidity_analysis": risk_assessment.get('portfolio_risk', {}).get('liquidity_risk', {})
            },
            "alerts_and_recommendations": {
                "active_alerts": risk_assessment.get('alerts', []),
                "high_risk_assets": risk_assessment.get('overall_summary', {}).get('high_governance_risk_assets', []),
                "recommendations": generate_detailed_recommendations(risk_assessment)
            },
            "compliance_status": assess_compliance_status(risk_assessment),
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "assessment_timestamp": risk_assessment.get('assessment_timestamp'),
                "report_version": "1.0"
            }
        }
        
        # Save detailed risk report
        risk_report_file = output_dir / "risk_assessment_report.json"
        with open(risk_report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate human-readable summary
        summary_file = output_dir / "risk_summary.md"
        with open(summary_file, 'w') as f:
            f.write(generate_markdown_summary(report))
        
        logger.info(f"📋 Risk report saved to {risk_report_file}")
        logger.info(f"📄 Risk summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate risk report: {e}")

def generate_executive_summary(risk_assessment: Dict) -> Dict:
    """Generate executive summary of risk assessment"""
    
    overall_summary = risk_assessment.get('overall_summary', {})
    portfolio_risk = risk_assessment.get('portfolio_risk', {})
    
    # Key metrics
    portfolio_metrics = portfolio_risk.get('portfolio_metrics', {})
    risk_summary = portfolio_risk.get('risk_summary', {})
    
    return {
        "overall_risk_level": overall_summary.get('portfolio_risk_level', 'UNKNOWN'),
        "requires_immediate_attention": overall_summary.get('requires_attention', False),
        "key_metrics": {
            "sharpe_ratio": portfolio_metrics.get('sharpe_ratio', 0),
            "max_drawdown": portfolio_metrics.get('max_drawdown', 0),
            "portfolio_volatility": portfolio_metrics.get('volatility', 0),
            "var_95": portfolio_metrics.get('var_95', 0)
        },
        "risk_counts": {
            "total_alerts": overall_summary.get('total_alerts', 0),
            "high_governance_risk_assets": len(overall_summary.get('high_governance_risk_assets', [])),
            "critical_alerts": len([a for a in risk_assessment.get('alerts', []) if a.get('level') == 'CRITICAL'])
        },
        "top_concerns": identify_top_concerns(risk_assessment)
    }

def identify_top_concerns(risk_assessment: Dict) -> List[str]:
    """Identify top risk concerns for executive summary"""
    
    concerns = []
    
    # Check portfolio-level risks
    portfolio_metrics = risk_assessment.get('portfolio_risk', {}).get('portfolio_metrics', {})
    risk_summary = risk_assessment.get('portfolio_risk', {}).get('risk_summary', {})
    
    if portfolio_metrics.get('sharpe_ratio', 0) < 0.5:
        concerns.append("Low risk-adjusted returns (Sharpe ratio < 0.5)")
    
    if abs(portfolio_metrics.get('max_drawdown', 0)) > 0.2:
        concerns.append("High maximum drawdown risk (>20%)")
    
    # Check concentration risk
    concentration = risk_assessment.get('portfolio_risk', {}).get('concentration_risk', {})
    if concentration.get('risk_level') in ['HIGH', 'CRITICAL']:
        concerns.append("High portfolio concentration risk")
    
    # Check governance risks
    high_gov_risk = risk_assessment.get('overall_summary', {}).get('high_governance_risk_assets', [])
    if len(high_gov_risk) > 2:
        concerns.append(f"Multiple assets with high governance risk ({len(high_gov_risk)} assets)")
    
    # Check critical alerts
    critical_alerts = [a for a in risk_assessment.get('alerts', []) if a.get('level') == 'CRITICAL']
    if critical_alerts:
        concerns.append(f"Critical risk alerts requiring immediate attention ({len(critical_alerts)} alerts)")
    
    return concerns[:5]  # Top 5 concerns

def generate_detailed_recommendations(risk_assessment: Dict) -> List[Dict]:
    """Generate detailed risk mitigation recommendations"""
    
    recommendations = []
    
    # Portfolio-level recommendations
    portfolio_metrics = risk_assessment.get('portfolio_risk', {}).get('portfolio_metrics', {})
    
    if portfolio_metrics.get('sharpe_ratio', 0) < 0.5:
        recommendations.append({
            "category": "Performance",
            "priority": "HIGH",
            "recommendation": "Improve risk-adjusted returns",
            "details": "Consider rebalancing to higher-quality assets or implementing momentum strategies",
            "timeline": "1-2 weeks"
        })
    
    if abs(portfolio_metrics.get('max_drawdown', 0)) > 0.15:
        recommendations.append({
            "category": "Risk Management",
            "priority": "HIGH", 
            "recommendation": "Implement downside protection",
            "details": "Consider adding defensive assets or implementing stop-loss mechanisms",
            "timeline": "Immediate"
        })
    
    # Asset-specific recommendations
    high_gov_risk = risk_assessment.get('overall_summary', {}).get('high_governance_risk_assets', [])
    if high_gov_risk:
        recommendations.append({
            "category": "Governance",
            "priority": "MEDIUM",
            "recommendation": f"Review high governance risk positions: {', '.join(high_gov_risk)}",
            "details": "Consider reducing exposure or implementing enhanced monitoring",
            "timeline": "1 week"
        })
    
    # Concentration recommendations
    concentration = risk_assessment.get('portfolio_risk', {}).get('concentration_risk', {})
    if concentration.get('risk_level') in ['HIGH', 'CRITICAL']:
        recommendations.append({
            "category": "Diversification",
            "priority": "MEDIUM",
            "recommendation": "Reduce portfolio concentration",
            "details": f"HHI of {concentration.get('hhi', 0):.3f} indicates high concentration. Consider adding more assets or rebalancing weights.",
            "timeline": "2-3 weeks"
        })
    
    return recommendations

def assess_compliance_status(risk_assessment: Dict) -> Dict:
    """Assess compliance with risk management policies"""
    
    compliance_checks = {
        "position_size_limits": True,
        "sector_concentration": True,
        "governance_standards": True,
        "liquidity_requirements": True,
        "risk_budget_adherence": True
    }
    
    violations = []
    
    # Check position size limits (15% max)
    concentration = risk_assessment.get('portfolio_risk', {}).get('concentration_risk', {})
    if concentration.get('max_weight', 0) > 0.15:
        compliance_checks["position_size_limits"] = False
        violations.append(f"Position size limit exceeded: {concentration.get('max_weight', 0)*100:.1f}% > 15%")
    
    # Check governance standards
    high_gov_risk = risk_assessment.get('overall_summary', {}).get('high_governance_risk_assets', [])
    if len(high_gov_risk) > 0:
        compliance_checks["governance_standards"] = False
        violations.append(f"Assets with high governance risk: {', '.join(high_gov_risk)}")
    
    # Check liquidity requirements
    liquidity = risk_assessment.get('portfolio_risk', {}).get('liquidity_risk', {})
    if liquidity.get('portfolio_liquidity', 1) < 0.5:
        compliance_checks["liquidity_requirements"] = False
        violations.append(f"Portfolio liquidity below threshold: {liquidity.get('portfolio_liquidity', 0)*100:.1f}% < 50%")
    
    overall_compliance = all(compliance_checks.values())
    
    return {
        "overall_compliant": overall_compliance,
        "compliance_checks": compliance_checks,
        "violations": violations,
        "compliance_score": sum(compliance_checks.values()) / len(compliance_checks)
    }

def generate_markdown_summary(report: Dict) -> str:
    """Generate human-readable markdown summary"""
    
    exec_summary = report["executive_summary"]
    compliance = report["compliance_status"]
    
    md = f"""# Portfolio Risk Assessment Summary

Generated: {report['report_metadata']['generated_at']}

## Executive Summary

**Overall Risk Level:** {exec_summary['overall_risk_level']}
**Requires Attention:** {"🚨 YES" if exec_summary['requires_immediate_attention'] else "✅ NO"}
**Compliance Status:** {"❌ NON-COMPLIANT" if not compliance['overall_compliant'] else "✅ COMPLIANT"}

### Key Metrics
- **Sharpe Ratio:** {exec_summary['key_metrics']['sharpe_ratio']:.3f}
- **Max Drawdown:** {exec_summary['key_metrics']['max_drawdown']*100:.1f}%
- **Portfolio Volatility:** {exec_summary['key_metrics']['portfolio_volatility']*100:.1f}%
- **VaR (95%):** {exec_summary['key_metrics']['var_95']*100:.1f}%

### Risk Overview
- **Total Alerts:** {exec_summary['risk_counts']['total_alerts']}
- **Critical Alerts:** {exec_summary['risk_counts']['critical_alerts']}
- **High Governance Risk Assets:** {exec_summary['risk_counts']['high_governance_risk_assets']}

## Top Concerns
"""
    
    for i, concern in enumerate(exec_summary.get('top_concerns', []), 1):
        md += f"{i}. {concern}\n"
    
    md += "\n## Recommendations\n"
    
    for rec in report['alerts_and_recommendations']['recommendations']:
        priority = rec['priority']
        emoji = "🔴" if priority == "HIGH" else "🟡" if priority == "MEDIUM" else "🟢"
        md += f"### {emoji} {rec['category']} - {rec['recommendation']}\n"
        md += f"**Priority:** {priority} | **Timeline:** {rec['timeline']}\n"
        md += f"{rec['details']}\n\n"
    
    if compliance['violations']:
        md += "## Compliance Violations\n"
        for violation in compliance['violations']:
            md += f"- ❌ {violation}\n"
    
    md += f"\n## Compliance Score: {compliance['compliance_score']*100:.0f}%\n"
    
    return md

def main():
    """Main integration function"""
    logger.info("🚀 Starting Risk Management Integration")
    
    try:
        # Integrate risk management into portfolio
        result = integrate_risk_into_generator()
        
        if result:
            logger.info("✅ Risk management integration completed successfully")
            
            # Print summary
            if 'risk_assessment' in result:
                summary = result['risk_assessment']['overall_summary']
                logger.info(f"📊 Portfolio Risk Summary:")
                logger.info(f"   Risk Level: {summary.get('portfolio_risk_level', 'UNKNOWN')}")
                logger.info(f"   Total Alerts: {summary.get('total_alerts', 0)}")
                logger.info(f"   High Risk Assets: {len(summary.get('high_governance_risk_assets', []))}")
                logger.info(f"   Requires Attention: {summary.get('requires_attention', False)}")
        else:
            logger.error("❌ Risk management integration failed")
            
    except Exception as e:
        logger.error(f"💥 Integration failed: {e}")
        raise

if __name__ == "__main__":
    main()