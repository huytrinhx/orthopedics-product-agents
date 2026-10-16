"""Read-only view of the master catalog's Product Family > Tray hierarchy,
as mirrored into Postgres by backend/ingestion/seed_master_catalog.py.
Admin-only, same as the rest of catalog/document management (see
auth.dependencies.require_admin). Nothing here writes: families and trays
come from the master file, and display-name editing is its own ticket.
"""
from fastapi import APIRouter, Depends

from auth.dependencies import require_admin
from auth.repository import UserRecord
from catalog.models import ProductFamilyOut, TrayOut
from catalog.repository import list_families_with_trays

router = APIRouter()


@router.get("/product-families", response_model=list[ProductFamilyOut])
async def list_product_families(
    admin: UserRecord = Depends(require_admin),
) -> list[ProductFamilyOut]:
    return [
        ProductFamilyOut(
            id=family.id,
            name=family.name,
            display_name=family.display_name,
            trays=[
                TrayOut(id=tray.id, name=tray.name, display_name=tray.display_name)
                for tray in family.trays
            ],
        )
        for family in await list_families_with_trays()
    ]
