from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


FONT = 'HeiseiKakuGo-W5'
pdfmetrics.registerFont(UnicodeCIDFont(FONT))


def _safe(value: Any) -> str:
    text = str(value or '').strip()
    return (
        text.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('\n', '<br/>')
    )


def create_lost_receipt_pdf(data: dict[str, Any]) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title='証憑紛失理由書（下書き）',
        author='人事アシスタントAI',
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        'JapaneseTitle', parent=styles['Title'], fontName=FONT,
        fontSize=20, leading=26, alignment=TA_CENTER, textColor=colors.HexColor('#202020'),
    )
    normal = ParagraphStyle(
        'JapaneseNormal', parent=styles['BodyText'], fontName=FONT,
        fontSize=10, leading=16, textColor=colors.HexColor('#262626'),
    )
    small = ParagraphStyle(
        'JapaneseSmall', parent=normal, fontSize=8.5, leading=13,
        textColor=colors.HexColor('#606060'),
    )
    right = ParagraphStyle('JapaneseRight', parent=normal, alignment=TA_RIGHT)
    label_style = ParagraphStyle(
        'JapaneseLabel', parent=normal, fontSize=9, textColor=colors.HexColor('#404040'),
    )

    story = [
        Paragraph('証憑紛失理由書', title),
        Paragraph('DRAFT / 下書き・未提出', ParagraphStyle(
            'Draft', parent=small, alignment=TA_CENTER,
            textColor=colors.HexColor('#B45309'), spaceBefore=2, spaceAfter=10,
        )),
        Paragraph(f'作成日: {_safe(data[created_date])}', right),
        Spacer(1, 5 * mm),
    ]

    rows = [
        ('所属部署', data['department']),
        ('申請者氏名', data['applicant_name']),
        ('利用日・購入日', data['expense_date']),
        ('金額', f'¥{int(data[amount]):,}'),
        ('支払先', data['payee']),
        ('経費の目的', data['purpose']),
    ]
    table_data = [
        [Paragraph(_safe(label), label_style), Paragraph(_safe(value), normal)]
        for label, value in rows
    ]
    summary_table = Table(table_data, colWidths=[38 * mm, 112 * mm])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F1EC')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#B8B5AE')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.extend([summary_table, Spacer(1, 6 * mm)])

    for heading, value in [
        ('紛失の経緯・理由', data['loss_reason']),
        ('添付する代替証憑', data['substitute_evidence']),
        ('再発防止策', data['prevention_plan']),
    ]:
        box = Table([
            [Paragraph(_safe(heading), label_style)],
            [Paragraph(_safe(value), normal)],
        ], colWidths=[150 * mm])
        box.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), FONT),
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#F3F1EC')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#B8B5AE')),
            ('LINEBELOW', (0, 0), (0, 0), 0.5, colors.HexColor('#B8B5AE')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('MINROWHEIGHT', (0, 1), (0, 1), 18 * mm),
        ]))
        story.extend([KeepTogether(box), Spacer(1, 4 * mm)])

    approval = Table([
        [Paragraph('申請者確認', label_style), Paragraph('直属上長承認', label_style), Paragraph('経理責任者承認', label_style)],
        ['', '', ''],
    ], colWidths=[50 * mm] * 3, rowHeights=[8 * mm, 20 * mm])
    approval.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F3F1EC')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#B8B5AE')),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.extend([
        Spacer(1, 3 * mm), approval, Spacer(1, 5 * mm),
        Paragraph(
            '注意: 本書はAIが入力内容から作成した下書きです。内容を確認し、カード明細等の代替資料を添付したうえで、社内規程に従って承認を受けてください。正式な提出・承認を自動的に行うものではありません。',
            small,
        ),
    ])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 8)
        canvas.setFillColor(colors.HexColor('#777777'))
        canvas.drawString(20 * mm, 10 * mm, 'ACMEテクノロジーズ株式会社 / 人事アシスタントAI生成')
        canvas.drawRightString(190 * mm, 10 * mm, f'{doc.page}')
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
