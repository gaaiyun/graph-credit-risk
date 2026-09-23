param([string]$docx)
$pdf = [System.IO.Path]::ChangeExtension($docx, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
  $doc = $word.Documents.Open($docx, $false, $true)
  $total = $doc.ComputeStatistics(2)
  $appx = -1
  foreach ($p in $doc.Paragraphs) { if ($p.Range.Text -like "附录*稳健性与复现*") { $appx = $p.Range.Information(3); break } }
  $doc.ExportAsFixedFormat($pdf, 17)
  $doc.Close($false)
  "TOTAL_PAGES=$total"
  "APPENDIX_STARTS_ON=$appx"
  "BODY_PAGES=$($appx - 1)"
  "PDF=$pdf"
} finally { $word.Quit() }
