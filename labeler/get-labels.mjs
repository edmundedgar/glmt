import { getLabelerLabelDefinitions } from "@skyware/labeler/scripts";

const credentials = {
  identifier: "label.goat.navy",
  password: process.env.APP_PASSWORD,
};

const defs = await getLabelerLabelDefinitions(credentials);
console.log(JSON.stringify(defs, null, 2));
