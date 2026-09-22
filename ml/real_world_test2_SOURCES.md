# Supplementary real-world test set — sources and status (ml/real_world_test2/)

The images themselves are kept out of git (`ml/real_world_test2/` is git-ignored, the same
policy as `ml/real_world_test/`). This file records where each image came from so the test is
reproducible by URL.

## Confirmed and downloaded
| file | truth class | source | license | status | model top-1 |
| --- | --- | --- | --- | --- | --- |
| rice/rice_bacterial_leaf_blight_1.jpg | Rice___Bacterial_leaf_blight | IRRI Photos (Flickr, via EOL) | CC BY-NC-SA 2.0 | downloaded, 2391x3600 (portrait — likely a whole plant/tiller, not a single-leaf close-up, so the framing does not match the app's expected input) | Rice___Leaf_blast @ 24.8% (WRONG; low confidence -> gated at 0.95) |

URL: https://farm6.staticflickr.com/5172/5575804225_2e2c7f2292_o.jpg

## Confirmed but not yet downloaded
| file | truth class | source | license | status |
| --- | --- | --- | --- | --- |
| wheat/wheat_yellow_rust_1.jpg | Wheat___yellow_rust | USDA-ARS via Wikimedia Commons (File:Stripe_rust.jpg) | Public domain | blocked by Wikimedia HTTP 429 at download time — retry later, or download manually and drop it in |

URL: https://upload.wikimedia.org/wikipedia/commons/0/0a/stripe_rust.jpg

## Scoring
`ml/score_real_world2.py` scores whatever is in `real_world_test2/` through the shipped
`agroveyra_model.tflite` with the app's own preprocessing and writes `ml/real_world_test2_results.txt`.
It deliberately writes its own results file so it never clobbers the original 16-photo
`ml/real_world_results.csv/.txt`.
