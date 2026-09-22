# Supplementary real-world test set — sources and status (ml/real_world_test2/)

The images themselves are kept out of git (`ml/real_world_test2/` is git-ignored, the same
policy as `ml/real_world_test/`). This file records where each image came from so the test is
reproducible by URL.

## Confirmed and downloaded
| file | truth class | source | license | status | model top-1 |
| --- | --- | --- | --- | --- | --- |
| rice/rice_bacterial_leaf_blight_1.jpg | Rice___Bacterial_leaf_blight | IRRI Photos (Flickr, via EOL) | CC BY-NC-SA 2.0 | downloaded, 2391x3600 (portrait — likely a whole plant/tiller, not a single-leaf close-up, so the framing does not match the app's expected input) | Rice___Leaf_blast @ 24.8% (WRONG; low confidence -> gated at 0.95) |

URL: https://farm6.staticflickr.com/5172/5575804225_2e2c7f2292_o.jpg
Attribution page: https://www.eol.org/media/14007621
Truth label: as-labelled by source (real field photo of an infected rice leaf; not verified pixel-by-pixel)

## Confirmed but not yet downloaded
| file | truth class | source | license | status |
| --- | --- | --- | --- | --- |
| wheat/wheat_yellow_rust_1.jpg | Wheat___yellow_rust | USDA-ARS via Wikimedia Commons (File:Stripe_rust.jpg) | Public domain (PD-USGOV-USDA-ARS) | blocked by Wikimedia HTTP 429 from this machine — download in a browser or from another network and drop it in, then re-score |

URL: https://upload.wikimedia.org/wikipedia/commons/0/0a/stripe_rust.jpg
Commons page: https://commons.wikimedia.org/wiki/File:Stripe_rust.jpg

## Unresolved (6 of 8) — not web-searchable to a confirmable direct URL
| file | truth class |
| --- | --- |
| cotton/cotton_bacterial_blight_1.jpg | Cotton___bacterial_blight |
| cotton/cotton_healthy_1.jpg | Cotton___healthy |
| wheat/wheat_healthy_1.jpg | Wheat___healthy |
| rice/rice_healthy_1.jpg | Rice___healthy |
| tomato/tomato_early_blight_1.jpg | Tomato___Early_blight |
| tomato/tomato_healthy_1.jpg | Tomato___healthy |

These need a *real single-leaf field close-up* with a *clear open license*; web search did not yield
one. Recommended source: phone photos (1 diseased + 1 healthy per crop, leaf filling 60-90% of frame)
— the actual deployment domain, zero licensing question.

Partial lead (unclear license, internal-use-only, not for publication):
Cotton bacterial blight close-up — Crop Protection Network, photos by Travis Faske
https://cropprotectionnetwork.org/encyclopedia/bacterial-blight-of-cotton

## Scoring
`ml/score_real_world2.py` scores whatever is in `real_world_test2/` through the shipped
`agroveyra_model.tflite` with the app's own preprocessing and writes `ml/real_world_test2_results.txt`.
It deliberately writes its own results file so it never clobbers the original 16-photo
`ml/real_world_results.csv/.txt`.
