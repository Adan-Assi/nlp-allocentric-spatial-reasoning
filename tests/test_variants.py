from src.constraints.underspec_constraints import generate_variants_for_text

text = "It's on the north side of Liberty Street a couple of blocks before the street ends. Across the street is a church."
vars_ = generate_variants_for_text(text, enabled_types=["direction", "radius", "proximity"])

for v in vars_:
    print("DROP:", v["dropped_types"], "\n", v["variant_text"], "\n---")